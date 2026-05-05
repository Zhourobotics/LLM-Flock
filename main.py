"""
Main entry point for the LLMs for Flocking simulation.

This file has been refactored to use a modular architecture where the main function
is simply an orchestrator that delegates work to specialized modules.
"""

import asyncio
import logging

from arguments import parse_cli_arguments
from simulation_runner import SimulationRunner

# Configure Logging for Performance Monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("performance.log", encoding="utf-8"),  # Save logs to a file
        logging.StreamHandler(),  # Print logs in console
    ],
)

logger = logging.getLogger(__name__)
args = parse_cli_arguments()


async def main():
    """Main function to manage the multi-agent system and simulation."""
    runner = SimulationRunner(args)
    
    # Initialize the simulation
    success = await runner.initialize_simulation()
    if not success:
        return
    
    # Run the simulation or plot results
    if args.mode == "run":
        await runner.run_simulation()
    
    # Plot results (both for run and plot modes)
    runner.plot_results()


if __name__ == "__main__":
    asyncio.run(main())