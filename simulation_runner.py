"""
Simulation Runner Module

This module orchestrates the entire simulation process, managing initialization,
execution, and result saving. It acts as the main controller that coordinates
all other modules.
"""

import os
import random
import time
import logging

from agents import Agent, Target
from data import Data
from graph import Graph
from plan_manager import PlanManager
from simulation_engine import SimulationEngine
from status_bar import status_bar


logger = logging.getLogger(__name__)


class SimulationRunner:
    """Main orchestrator for running multi-agent flocking simulations."""
    
    def __init__(self, args):
        self.args = args
        self.agents = []
        self.targets = []
        self.plan_manager = PlanManager(args)
        self.simulation_engine = SimulationEngine(args)
        self.results_path = f"./results/{args.name}/results"
    
    async def initialize_simulation(self):
        """Initialize all components needed for the simulation."""
        # Set logging level based on debug_mode
        log_level = logging.DEBUG if self.args.debug_mode else logging.INFO
        logging.getLogger().setLevel(log_level)

        logger.info(f"############# Starting simulation: {self.args.name} #############")
        
        # Check if test already exists
        if self.args.mode == "run":
            if os.path.isfile(self.results_path):
                logger.error(f"Test {self.args.name} already exists! Aborting.")
                return False

            random.seed(self.args.seed)


            # Initialize Agents
            await self._initialize_agents()

            # Initialize Targets
            self._initialize_targets()

        return True


    async def _initialize_agents(self):
        """Initialize all agents for the simulation."""
        self.agents = [
            Agent(
                identifier=i,
                # position randomize if active agent, else use predefined position
                position=(
                    [
                        round(random.uniform(self.args.spawn_x_min, self.args.spawn_x_max), 2),
                        round(random.uniform(self.args.spawn_y_min, self.args.spawn_y_max), 2),
                    ]
                    if (i >= self.args.zombies and self.args.randomize)
                    else self.args.initial_position[i]
                ),
                active=(i >= self.args.zombies),
            )
            for i in range(self.args.agents)
        ]

        # Initialize plan source tracking for each agent
        for agent in self.agents:
            # Initially, each agent is its own plan source
            agent.plan_source_id = agent.identifier
            agent.plan_origin_id = agent.identifier

    def _initialize_targets(self):
        """Initialize all targets for the simulation."""
        self.targets = [
            Target(
                identifier=i,
                position=self.args.target_position[i],
                active=self.args.target_active,
                speed=self.args.target_speed,
                dt=self.args.dt,
            )
            for i in range(self.args.targets)
        ]

    async def run_simulation(self):
        """Execute the main simulation loop."""
        if self.args.mode != "run":
            return

        try:
            # Start status bar
            status_bar.start()
            status_bar.update(self.args.name, 0, self.args.rounds, "Initializing")

            # Generate initial plans if not in basic mode
            await self._generate_initial_plans()

            # Execute simulation rounds
            await self._execute_simulation_rounds()

            # Log final results
            self._log_final_results()

            # Save data
            status_bar.update(self.args.name, self.args.rounds, self.args.rounds, "Saving")
            Data.save(self.agents, self.args, identifier=self.args.name)
            
            # Final status
            status_bar.update(self.args.name, self.args.rounds, self.args.rounds, "Complete")

        except Exception as e:
            logger.error(f"Simulation terminated unexpectedly: {e}")
            # Save partial results using current round count
            Data.save(self.agents, self.args, identifier=self.args.name, last_round=getattr(self, '_current_round', 0) + 1)
            raise
        finally:
            # Clean up status bar
            status_bar.cleanup()

    async def _generate_initial_plans(self):
        """Generate initial plans for all agents if not in basic mode."""
        if self.args.agent_mode == "basic":
            logger.info(
                "Running in basic mode - no planning phase, using only formation instructions"
            )
        else:
            # Generate plans for all other modes
            if self.args.agent_mode == "influence":
                logger.info(
                    "Running in influence mode - agents can adopt plans from more influential neighbors"
                )
            elif self.args.agent_mode == "plan":
                logger.info(
                    "Running in plan mode - all agents follow agent 0's plan with index matching their ID"
                )
            else:
                logger.info(
                    "Running in naive mode - agents will generate and execute their own plans independently"
                )

            # Parallel plan generation for all modes
            await self.plan_manager.generate_plan(self.agents)

    async def _execute_simulation_rounds(self):
        """Execute all simulation rounds."""
        for r in range(self.args.rounds):
            self._current_round = r  # Store current round for error handling

            # Update status bar
            status_bar.update(self.args.name, r + 1, self.args.rounds, "Parallel")
            
            logger.info(f"====== ROUND {r + 1}/{self.args.rounds} ======")
            tick = time.time()

            # Handle plan updates for influence and plan modes
            if self.args.agent_mode in ["influence", "plan"]:
                # Calculate influence for influence mode
                influences = (
                    self.simulation_engine.calculate_influence(self.agents)
                    if self.args.agent_mode == "influence"
                    else {}
                )
                # Update plans based on mode - THIS HAPPENS EVERY ROUND
                self.plan_manager.update_plans_based_on_influence(self.agents, influences)

            # Execute agent movement in parallel for all agents
            await self.simulation_engine.parallel_step(self.agents)

            time_lapse = time.time() - tick
            logger.info(f"Time for this round: {time_lapse:.2f} seconds")

    def _log_final_results(self):
        """Log final drone positions."""
        logger.info("Final Drone Positions:")
        for agent in self.agents:
            logger.info(f"Agent {agent.identifier}: {agent.position_history}")

    def plot_results(self):
        """Plot animation results if data exists."""
        if os.path.isfile(self.results_path):
            Graph.plot_animated(Data.load(self.args))
        else:
            logger.error(f"Test {self.args.name} does not exist! Cannot plot results.")
