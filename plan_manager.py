"""
Plan Manager Module

This module handles all planning-related functionality including plan generation,
influence-based plan adoption, leader selection, and waypoint assignment.
"""

import asyncio
import logging
import numpy as np

from api_client import retry_api_call
from structured_parser import parse_plan_response
from structured_models import PlanGenerationResponse
from prompts import Prompt

# ANSI color codes for logging
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

logger = logging.getLogger(__name__)


def calculate_distance(position1, position2):
    """Calculate the Euclidean distance between two positions."""
    return np.linalg.norm(np.array(position1) - np.array(position2))


class PlanManager:
    """Manages all planning operations for the multi-agent system."""
    
    def __init__(self, args):
        self.args = args
    
    async def generate_plan(self, agents):
        """Generate plans for all agents in parallel."""
        # Generate plans in parallel for all models
        await self._generate_plan_parallel(agents)
    
    
    async def _generate_plan_parallel(self, agents):
        """Generate plans in parallel for all models."""
        coroutines = []
        for agent in agents:
            # Feed the latest position, prepare latest message
            message = Prompt.plan_inquiry.format(
                self.args.agents, self.args.formation, self.args.desired_distance, self.args.agents
            )

            # Send the message to LLM with structured output format
            coroutines.append(
                retry_api_call(
                    agent,
                    message,
                    self.args.model,
                    self.args.memory_limit,
                    response_format=PlanGenerationResponse
                )
            )

        try:
            # Wait for the response from LLM
            results = await asyncio.gather(*coroutines)

            # Validate and update agent positions
            for agent, result in zip(agents, results):
                if result:
                    # Parse plan data from structured JSON response.
                    plan_data, my_plan_idx, plan_reasoning = parse_plan_response(result)

                    agent.plan = plan_data
                    agent.my_plan = my_plan_idx

                    if agent.plan and agent.my_plan is not None and 0 <= agent.my_plan < len(agent.plan):
                        # location of the plan
                        agent.my_goal = agent.plan[agent.my_plan]
                        agent.plan_subscriber = 0

                        # Ensure plan_source_id and plan_origin_id are set to the agent's own ID
                        # when they generate their own plan - this is crucial for coloring
                        agent.plan_source_id = agent.identifier
                        agent.plan_origin_id = agent.identifier

                        logger.info(
                            f"Agent {agent.identifier} | Plan: {agent.plan} | My Plan: {agent.my_goal} | "
                            f"My Plan Index: {agent.my_plan} | Plan Origin: {agent.plan_origin_id}"
                            + (f" | Reasoning: {plan_reasoning}" if plan_reasoning else "")
                        )
                    else:
                        logger.warning(f"Agent {agent.identifier} | Invalid plan or plan index")

        except Exception as e:
            logger.error(f"Error during agent decision-making: {e}")
    
    def update_plans_based_on_influence(self, agents, influences):
        """
        Update the agents' plans based on their influence and plan popularity.
        Multiple modes supported:
        - influence: dynamically determine leader based on influence and followers
        - plan: all agents follow agent 0's plan with index based on agent ID
        """
        # Print debug info before plan updates
        self._print_plan_sources(agents, "BEFORE")

        if self.args.agent_mode == "plan":
            self._update_plans_with_agent0_as_leader(agents)
            return

        if self.args.agent_mode == "influence":
            plan_usage, assigned_waypoints = self._compute_plan_usage_and_assignments(agents)

            if any(agent.plan is not None for agent in agents):
                cluster_leaders, clusters = self._select_local_leaders(agents)
                if cluster_leaders:
                    self._process_local_leaders_influence(
                        agents, cluster_leaders, clusters, influences, assigned_waypoints
                    )

                # Second pass: pairwise influence comparison between all communicating agents
                self._process_pairwise_influence(
                    agents, influences, plan_usage, assigned_waypoints
                )

        # Print debug info after plan updates
        self._print_plan_sources(agents, "AFTER")

    def _print_plan_sources(self, agents, phase):
        """Log plan sources and (after updates) record current origin per agent."""
        logger.debug("=== PLAN SOURCES %s UPDATE ===", phase)
        for i, agent in enumerate(agents):
            logger.debug(
                "Agent %d: plan_origin=%s, plan_source=%s, following %s",
                i,
                agent.plan_origin_id,
                agent.plan_source_id,
                agent.parent_agent.identifier if agent.parent_agent else "None",
            )
            if phase == "AFTER":
                agent.plan_origin.append(agent.plan_origin_id)

    def _update_plans_with_agent0_as_leader(self, agents):
        """Implement 'plan' mode where agent 0 is the designated leader."""
        # Get agent 0's plan
        leader = next((a for a in agents if a.identifier == 0), None)
        if leader and leader.plan is not None:
            # Log the leader's plan
            logger.info(f"{GREEN}Using Agent 0's plan in 'plan' mode: {leader.plan}{RESET}")

            # Assign the plan to each agent based on their ID
            for agent in agents:
                if agent != leader and (
                    agent.plan != leader.plan or agent.plan_origin_id != leader.identifier
                ):
                    # Before changing plan, log the transition
                    logger.info(
                        f"{YELLOW}Agent {agent.identifier} changing plan: origin {agent.plan_origin_id} -> {leader.identifier}{RESET}"
                    )

                    agent.plan = leader.plan
                    # Set my_plan index to match agent ID (with bounds checking)
                    plan_index = min(agent.identifier, len(leader.plan) - 1)
                    agent.my_plan = plan_index
                    agent.my_goal = leader.plan[plan_index]
                    agent.parent_agent = leader
                    # Track the leader as the plan source and origin
                    agent.plan_source_id = leader.identifier
                    agent.plan_origin_id = leader.identifier
                    logger.info(
                        f"{CYAN}Agent {agent.identifier} adopted Agent 0's plan with index {plan_index}: {agent.my_goal}{RESET}"
                    )

    def _compute_plan_usage_and_assignments(self, agents):
        """Compute plan usage counts and track assigned waypoints."""
        plan_usage = {}
        assigned_waypoints = {}

        for agent in agents:
            if agent.plan is not None:
                # Use the plan_origin_id as a key component for the plan
                origin_id = getattr(agent, "plan_origin_id", agent.identifier)
                plan_key = (origin_id, tuple(tuple(coord) for coord in agent.plan))
                plan_usage[plan_key] = plan_usage.get(plan_key, 0) + 1

                # Initialize assigned waypoints tracking for this plan
                if plan_key not in assigned_waypoints:
                    assigned_waypoints[plan_key] = set()

                # Mark this agent's waypoint as assigned
                if agent.my_plan is not None:
                    assigned_waypoints[plan_key].add(agent.my_plan)

        return plan_usage, assigned_waypoints

    def _select_local_leaders(self, agents):
        """Select local leaders for each connected sub-group based on communication range."""
        # Dictionary to track which cluster each agent belongs to
        clusters = {}
        cluster_id = 0

        # First, identify connected clusters using a simple graph traversal
        unassigned = set(range(len(agents)))

        while unassigned:
            # Start a new cluster
            node_id = next(iter(unassigned))
            frontier = [node_id]
            current_cluster = set()

            # Find all connected agents (BFS)
            while frontier:
                current = frontier.pop(0)
                if current in current_cluster:
                    continue

                current_cluster.add(current)
                unassigned.remove(current)

                # Find neighbors within communication range
                for neighbor_id in unassigned:
                    if (
                        calculate_distance(
                            agents[current].position, agents[neighbor_id].position
                        )
                        <= self.args.comm_range
                    ):
                        frontier.append(neighbor_id)

            # Assign all agents in this connected component to the same cluster
            for agent_id in current_cluster:
                clusters[agent_id] = cluster_id

            cluster_id += 1

        # Now find a leader for each cluster
        cluster_leaders = {}
        for cluster_num in range(cluster_id):
            # Get all agents in this cluster
            cluster_agents = [
                agents[i] for i in range(len(agents)) if clusters.get(i) == cluster_num
            ]
            cluster_agents_with_plans = [a for a in cluster_agents if a.plan is not None]

            if cluster_agents_with_plans:
                # Select leader based on influence, subscribers, and ID
                leader = max(
                    cluster_agents_with_plans,
                    key=lambda a: (a.influence, a.plan_subscriber, -a.identifier),
                )
                cluster_leaders[cluster_num] = leader
                logger.info(
                    f"{GREEN}Agent {leader.identifier} selected as leader for cluster {cluster_num} "
                    f"(influence: {leader.influence}, subscribers: {leader.plan_subscriber}){RESET}"
                )

        return cluster_leaders, clusters

    def _process_local_leaders_influence(self, agents, cluster_leaders, clusters, influences, assigned_waypoints):
        """Process local leaders' influence on nearby agents within their respective clusters."""
        adoption_count = 0

        for cluster_num, leader in cluster_leaders.items():
            leader_origin_id = getattr(leader, "plan_origin_id", leader.identifier)
            leader_plan_key = (
                leader_origin_id,
                tuple(tuple(coord) for coord in leader.plan),
            )

            # Influence agents in this cluster
            for i, other_agent in enumerate(agents):
                # Only consider agents in the same cluster
                if (
                    clusters.get(i) != cluster_num
                    or other_agent.identifier == leader.identifier
                ):
                    continue

                # Check if already using the leader's plan
                other_origin_id = getattr(
                    other_agent, "plan_origin_id", other_agent.identifier
                )
                if other_agent.plan != leader.plan or other_origin_id != leader_origin_id:
                    # Before changing plan, log the transition
                    logger.info(
                        f"{YELLOW}Agent {other_agent.identifier} changing plan: origin {other_agent.plan_origin_id} -> {leader_origin_id} (cluster {cluster_num}){RESET}"
                    )

                    # Adopt the leader's plan
                    other_agent.plan = leader.plan
                    self._assign_waypoint(
                        other_agent, leader.plan, leader_plan_key, assigned_waypoints
                    )
                    other_agent.parent_agent = leader
                    other_agent.plan_source_id = leader.identifier
                    other_agent.plan_origin_id = leader_origin_id
                    leader.plan_subscriber += 1
                    adoption_count += 1

                    logger.info(
                        f"{BLUE}Agent {other_agent.identifier} adopted cluster {cluster_num} leader's (Agent {leader.identifier}) plan "
                        f"with origin {leader_origin_id}, waypoint {other_agent.my_plan}{RESET}"
                    )

        if adoption_count == 0:
            logger.info(f"{YELLOW}No agents adopted any leader's plan this round{RESET}")

        return adoption_count

    def _process_pairwise_influence(self, agents, influences, plan_usage, assigned_waypoints):
        """Process pairwise influence comparisons between all agent pairs."""
        for agent in agents:
            for other_agent in agents:
                if agent.identifier != other_agent.identifier:
                    distance = calculate_distance(agent.position, other_agent.position)
                    if distance <= self.args.comm_range:
                        # Only consider plan changes if plans are different or origin is different
                        agent_origin_id = getattr(agent, "plan_origin_id", agent.identifier)
                        other_origin_id = getattr(
                            other_agent, "plan_origin_id", other_agent.identifier
                        )

                        if (
                            agent.plan == other_agent.plan
                            and agent_origin_id == other_origin_id
                        ):
                            continue

                        # Evaluate plan popularity and influence
                        should_adopt_plan = self._evaluate_plan_adoption(
                            agent,
                            other_agent,
                            agent_origin_id,
                            other_origin_id,
                            influences,
                            plan_usage,
                        )

                        if should_adopt_plan:
                            # Process plan adoption
                            self._adopt_plan(
                                agent, other_agent, agent_origin_id, assigned_waypoints
                            )

    def _evaluate_plan_adoption(self, agent, other_agent, agent_origin_id, other_origin_id, influences, plan_usage):
        """Evaluate whether other_agent should adopt agent's plan based on influence and popularity."""
        agent_plan_key = (
            (
                agent_origin_id,
                tuple(tuple(coord) for coord in agent.plan),
            )
            if agent.plan is not None
            else None
        )
        other_plan_key = (
            (
                other_origin_id,
                tuple(tuple(coord) for coord in other_agent.plan),
            )
            if other_agent.plan is not None
            else None
        )
        agent_plan_usage = plan_usage.get(agent_plan_key, 0)
        other_plan_usage = plan_usage.get(other_plan_key, 0)

        return (
            influences[agent.identifier] > influences[other_agent.identifier]
            and agent_plan_usage > other_plan_usage
        )

    def _adopt_plan(self, agent, other_agent, agent_origin_id, assigned_waypoints):
        """Process plan adoption from agent to other_agent."""
        # Before changing plan, log the transition
        logger.info(
            f"{YELLOW}Agent {other_agent.identifier} changing plan: origin {other_agent.plan_origin_id} -> {agent_origin_id}{RESET}"
        )

        other_agent.plan = agent.plan

        # Create plan key for waypoint assignment tracking
        agent_plan_key = (
            agent_origin_id,
            tuple(tuple(coord) for coord in agent.plan),
        )

        # Assign waypoint to the agent
        self._assign_waypoint(other_agent, agent.plan, agent_plan_key, assigned_waypoints)

        # Update parent agent relationship with better tracking
        self._update_parent_relationship(agent, other_agent, agent_origin_id)

        logger.info(
            f"{CYAN}Agent {other_agent.identifier} adopted plan from Agent {agent.identifier} "
            f"(origin: Agent {other_agent.plan_origin_id}) with waypoint {other_agent.my_plan}{RESET}"
        )

    def _assign_waypoint(self, agent, plan, plan_key, assigned_waypoints):
        """Assign an appropriate waypoint from the plan to the agent."""
        # Find an available waypoint in the adopted plan
        available_waypoints = []
        for i in range(len(plan)):
            if i not in assigned_waypoints.get(plan_key, set()):
                available_waypoints.append(i)

        # If no available waypoints, use agent's ID (bounded by plan size)
        if not available_waypoints:
            agent.my_plan = min(agent.identifier, len(plan) - 1)
            logger.info(
                f"{YELLOW}No available waypoints in plan - using ID-based assignment{RESET}"
            )
        else:
            # Choose closest available waypoint
            closest_waypoint = None
            min_distance = float("inf")
            for wp_idx in available_waypoints:
                dist = calculate_distance(agent.position, plan[wp_idx])
                if dist < min_distance:
                    min_distance = dist
                    closest_waypoint = wp_idx

            agent.my_plan = closest_waypoint
            # Mark this waypoint as assigned
            if plan_key not in assigned_waypoints:
                assigned_waypoints[plan_key] = set()
            assigned_waypoints[plan_key].add(closest_waypoint)

        agent.my_goal = plan[agent.my_plan]

    def _update_parent_relationship(self, agent, other_agent, agent_origin_id):
        """Update parent agent relationship with proper tracking of source and origin IDs."""
        if agent.parent_agent is None:
            # Agent is the plan originator
            agent.plan_subscriber += 1
            other_agent.parent_agent = agent
            # Agent is its own plan source and origin
            other_agent.plan_source_id = agent.identifier  # Direct source
            other_agent.plan_origin_id = agent_origin_id  # Original source
        else:
            # Agent is passing along someone else's plan
            agent.parent_agent.plan_subscriber += 1
            other_agent.parent_agent = agent.parent_agent
            # Track both immediate source and original origin
            other_agent.plan_source_id = agent.identifier  # Direct source
            other_agent.plan_origin_id = agent_origin_id  # Preserve original source
