"""
Simulation Engine Module

This module handles the core simulation mechanics including agent movement,
influence calculation, and step execution.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Set, Tuple

from api_client import retry_api_call
from collision_detection import check_collision
from structured_parser import parse_agent_response
from structured_models import AgentMoveResponse
from prompts import Prompt

logger = logging.getLogger(__name__)


def calculate_distance(position1, position2):
    """Calculate the Euclidean distance between two positions."""
    return np.linalg.norm(np.array(position1) - np.array(position2))


class SimulationEngine:
    """Handles core simulation mechanics and agent interactions."""

    def __init__(self, args):
        self.args = args
        self.max_collision_reprompts = max(
            0, int(getattr(args, "collision_reprompt_attempts", 2))
        )
        timeout = float(getattr(args, "agent_response_timeout", 120.0))
        self.agent_response_timeout = timeout if timeout > 0 else 120.0
        self.collision_buffer = max(0.0, float(getattr(args, "collision_buffer", 0.5)))

    @staticmethod
    def _get_recent_positions(agent, limit):
        """Return up to `limit` recent positions, ensuring current position is included."""
        history = list(agent.position_history) if agent.position_history else []
        current_position = list(agent.position)

        if not history or history[-1] != current_position:
            history.append(current_position)

        return history[-limit:] if limit > 0 else history

    def _get_comm_neighbors(self, agent, agents):
        """Return neighboring agents within current communication range."""
        comm_range = float(self.args.comm_range)
        neighbors = [
            other_agent
            for other_agent in agents
            if other_agent.identifier != agent.identifier
            and calculate_distance(agent.position, other_agent.position) <= comm_range
        ]
        return sorted(neighbors, key=lambda other_agent: other_agent.identifier)

    def _build_trajectory_context(self, agent, comm_neighbors):
        """Build trajectory-only history context for the current movement update."""
        limit = max(1, int(self.args.memory_limit))
        self_history = self._get_recent_positions(agent, limit)

        neighbor_lines = []
        for other_agent in comm_neighbors:
            other_history = self._get_recent_positions(other_agent, limit)
            neighbor_lines.append(
                f"- Agent {other_agent.identifier}: {other_history}"
            )

        neighbors_text = (
            "\n".join(neighbor_lines)
            if neighbor_lines
            else "- None within communication range"
        )
        return (
            f"Trajectory History (last {limit} points):\n"
            f"- Self (Agent {agent.identifier}): {self_history}\n"
            "Neighbor Trajectories (within communication range):\n"
            f"{neighbors_text}\n\n"
        )

    def _build_round_message(self, agent, agents):
        """Build the base movement prompt for one agent for this round."""
        comm_neighbors = self._get_comm_neighbors(agent, agents)
        other_agent_positions = ", ".join(
            str(a.position) for a in comm_neighbors
        )
        other_agent_goal = ", ".join(
            str(a.my_goal) for a in comm_neighbors
        )
        trajectory_context = self._build_trajectory_context(agent, comm_neighbors)

        if self.args.agent_mode == "basic":
            return (
                trajectory_context
                + Prompt.agent_positions.format(agent.position, other_agent_positions)
                + Prompt.agent_task
            )

        return (
            trajectory_context
            + Prompt.agent_positions.format(agent.position, other_agent_positions)
            + Prompt.plan_general.format(agent.plan, agent.my_goal)
            + Prompt.other_agent_goal.format(other_agent_goal)
            + Prompt.agent_task
        )

    @staticmethod
    def _agent_priority(agent) -> Tuple[float, int]:
        """
        Return priority tuple where higher values mean higher priority.
        Priority rule: urgency (distance to final goal) first, then lower id.
        """
        if agent.my_goal is None:
            urgency = 0.0
        else:
            try:
                urgency = float(
                    np.linalg.norm(
                        np.array(agent.my_goal, dtype=float)
                        - np.array(agent.position, dtype=float)
                    )
                )
            except Exception:
                urgency = 0.0

        return urgency, -int(agent.identifier)

    def _select_yield_agents(
        self, collision_pairs: List[Tuple[int, int]], agents
    ) -> Set[int]:
        """Select lower-priority agents to reprompt for each collision pair."""
        yield_agents: Set[int] = set()
        for first_idx, second_idx in collision_pairs:
            first_priority = self._agent_priority(agents[first_idx])
            second_priority = self._agent_priority(agents[second_idx])
            if first_priority >= second_priority:
                yield_agents.add(second_idx)
            else:
                yield_agents.add(first_idx)
        return yield_agents

    @staticmethod
    def _pair_min_distance(collision_info: Dict, first_idx: int, second_idx: int):
        min_idx, max_idx = min(first_idx, second_idx), max(first_idx, second_idx)
        key = f"{min_idx}-{max_idx}"
        return collision_info.get("trajectory_min_distances", {}).get(key)

    def _build_collision_reprompt_message(
        self,
        base_message: str,
        agent_idx: int,
        involved_indices: List[int],
        agents,
        start_positions: List[List[float]],
        candidate_positions: List[List[float]],
        collision_info: Dict,
        min_required_distance: float,
    ) -> str:
        """Append explicit safety feedback for collision-driven reprompting."""
        conflict_lines = []
        for other_idx in involved_indices:
            min_distance = self._pair_min_distance(collision_info, agent_idx, other_idx)
            distance_text = (
                f"{min_distance:.3f}" if isinstance(min_distance, (int, float)) else "unknown"
            )
            conflict_lines.append(
                f"- With Agent {agents[other_idx].identifier}: min predicted distance {distance_text}"
            )

        fixed_trajectory_lines = []
        for other_idx in involved_indices:
            fixed_trajectory_lines.append(
                f"- Agent {agents[other_idx].identifier}: {start_positions[other_idx]} -> {candidate_positions[other_idx]}"
            )

        reprompt_feedback = (
            "\n\nSafety Checker Feedback:\n"
            f"Your proposed trajectory {start_positions[agent_idx]} -> {candidate_positions[agent_idx]} "
            "is not collision-free.\n"
            f"Required minimum separation: {min_required_distance:.3f}\n"
            "Conflicts detected:\n"
            + "\n".join(conflict_lines)
            + "\nFixed trajectories to avoid:\n"
            + "\n".join(fixed_trajectory_lines)
            + "\nReturn a NEW next position that is collision-free while keeping the same JSON schema."
        )

        return base_message + reprompt_feedback

    async def _request_agent_move(self, agent, message: str):
        """Request one movement proposal with timeout and safe fallback behavior."""
        try:
            response = await asyncio.wait_for(
                retry_api_call(
                    agent,
                    message,
                    self.args.model,
                    self.args.memory_limit,
                    response_format=AgentMoveResponse,
                ),
                timeout=self.agent_response_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Agent {agent.identifier} | Timed out after {self.agent_response_timeout:.1f}s; holding current position."
            )
            return list(agent.position), None
        except Exception as exc:
            logger.warning(
                f"Agent {agent.identifier} | Failed to get movement response ({exc}); holding current position."
            )
            return list(agent.position), None

        if not response:
            logger.warning(
                f"Agent {agent.identifier} | No valid response after retries; holding current position."
            )
            return list(agent.position), None

        position, reasoning = parse_agent_response(response)
        if not position:
            logger.warning(
                f"Agent {agent.identifier} | Failed to parse JSON position; holding current position."
            )
            return list(agent.position), None

        return [float(position[0]), float(position[1])], reasoning

    def _detect_collisions(
        self, start_positions: List[List[float]], candidate_positions: List[List[float]]
    ) -> Tuple[bool, Dict]:
        """Run centralized collision checks for this round."""
        min_required_distance = float(self.args.safe_distance) + self.collision_buffer
        return check_collision(
            current_positions=candidate_positions,
            prev_positions=start_positions,
            min_distance=min_required_distance,
        )

    def _apply_emergency_hold(
        self,
        start_positions: List[List[float]],
        candidate_positions: List[List[float]],
    ) -> Tuple[bool, Dict, Set[int]]:
        """
        Deterministic final fallback: freeze all agents involved in remaining collisions.
        Iterates until stable so newly-created conflicts from freezing are also resolved.
        """
        held_indices: Set[int] = set()
        for _ in range(len(candidate_positions)):
            collision_detected, collision_info = self._detect_collisions(
                start_positions, candidate_positions
            )
            if not collision_detected:
                return False, collision_info, held_indices

            collision_pairs = collision_info.get("all_collision_pairs", [])
            to_hold = {idx for pair in collision_pairs for idx in pair} - held_indices
            if not to_hold:
                return True, collision_info, held_indices

            for idx in to_hold:
                candidate_positions[idx] = list(start_positions[idx])
            held_indices.update(to_hold)

        collision_detected, collision_info = self._detect_collisions(
            start_positions, candidate_positions
        )
        return collision_detected, collision_info, held_indices

    async def parallel_step(self, agents):
        """Execute one step for all agents in parallel."""
        # Execute in parallel for all models
        await self._parallel_step_parallel(agents)

    async def _parallel_step_parallel(self, agents):
        """Execute one parallel movement round with centralized collision gating."""
        if not agents:
            return

        base_messages = []
        for agent in agents:
            message = self._build_round_message(agent, agents)
            base_messages.append(message)
            logger.info(f"Agent {agent.identifier} - Position: {agent.position}")

        try:
            # Phase 1: collect one candidate move per agent.
            initial_tasks = [
                self._request_agent_move(agent, message)
                for agent, message in zip(agents, base_messages)
            ]
            initial_results = await asyncio.gather(*initial_tasks)

            start_positions = [list(agent.position) for agent in agents]
            candidate_positions = [list(position) for position, _ in initial_results]
            candidate_reasonings = [reasoning for _, reasoning in initial_results]

            min_required_distance = float(self.args.safe_distance) + self.collision_buffer
            collision_detected, collision_info = self._detect_collisions(
                start_positions, candidate_positions
            )

            # Phase 2: reprompt lower-priority agents involved in conflicts.
            for attempt in range(self.max_collision_reprompts):
                if not collision_detected:
                    break

                collision_pairs = collision_info.get("all_collision_pairs", [])
                yield_indices = sorted(
                    self._select_yield_agents(collision_pairs, agents)
                )
                if not yield_indices:
                    break

                logger.warning(
                    "Collision checker detected conflicts %s; reprompting agents %s (attempt %d/%d).",
                    collision_pairs,
                    [agents[idx].identifier for idx in yield_indices],
                    attempt + 1,
                    self.max_collision_reprompts,
                )

                reprompt_tasks = []
                reprompt_indices = []
                for idx in yield_indices:
                    involved_indices = sorted(
                        {
                            second_idx if first_idx == idx else first_idx
                            for first_idx, second_idx in collision_pairs
                            if idx in (first_idx, second_idx)
                        }
                    )
                    reprompt_message = self._build_collision_reprompt_message(
                        base_message=base_messages[idx],
                        agent_idx=idx,
                        involved_indices=involved_indices,
                        agents=agents,
                        start_positions=start_positions,
                        candidate_positions=candidate_positions,
                        collision_info=collision_info,
                        min_required_distance=min_required_distance,
                    )
                    reprompt_tasks.append(
                        self._request_agent_move(agents[idx], reprompt_message)
                    )
                    reprompt_indices.append(idx)

                reprompt_results = await asyncio.gather(*reprompt_tasks)
                for idx, (new_position, new_reasoning) in zip(
                    reprompt_indices, reprompt_results
                ):
                    candidate_positions[idx] = list(new_position)
                    candidate_reasonings[idx] = new_reasoning

                collision_detected, collision_info = self._detect_collisions(
                    start_positions, candidate_positions
                )

            # Phase 3: deterministic fallback if collisions still remain.
            if collision_detected:
                logger.warning(
                    "Collision checker still detects conflicts after reprompts; applying emergency hold fallback."
                )
                (
                    collision_detected,
                    collision_info,
                    held_indices,
                ) = self._apply_emergency_hold(start_positions, candidate_positions)
                if held_indices:
                    logger.warning(
                        "Emergency hold applied to agents %s.",
                        [agents[idx].identifier for idx in sorted(held_indices)],
                    )
                if collision_detected:
                    logger.error(
                        "Collision checker could not resolve all conflicts; initial state may already violate minimum separation."
                    )

            # Phase 4: commit safe (or fallback) positions.
            for idx, agent in enumerate(agents):
                agent.update(candidate_positions[idx])
                if candidate_reasonings[idx]:
                    logger.debug(
                        f"Agent {agent.identifier} reasoning: {candidate_reasonings[idx]}"
                    )

        except Exception as exc:
            logger.error(f"Error during agent decision-making: {exc}")

    def calculate_influence(self, agents):
        """
        Calculate the number of neighbors within the communication range and the influence for each agent.

        Args:
            agents (list): List of agent objects.

        Returns:
            dict: A dictionary with agent identifiers as keys and their influence as values.
        """
        influences = {}
        total_agents = self.args.agents

        for agent in agents:
            neighbors_within_range = 0
            for other_agent in agents:
                if agent.identifier != other_agent.identifier:
                    distance = calculate_distance(agent.position, other_agent.position)
                    if distance <= self.args.comm_range:
                        neighbors_within_range += 1  # count agent as neighbor
            # Normalize the neighbor count by dividing by total agent count
            influence = neighbors_within_range / total_agents
            agent.influence = influence
            influences[agent.identifier] = influence

        return influences
