import argparse
import yaml
import os

CONFIG_FILE = "config.yaml"


def load_config():
    """Load configuration from config.yaml and ensure it exists."""
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(
            f"Error: Configuration file '{CONFIG_FILE}' not found. Please create one."
        )

    print(f"Loading configuration from {CONFIG_FILE}...")
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


def parse_cli_arguments():
    """Parse CLI arguments and merge them with config.yaml settings (CLI args take priority)."""
    parser = argparse.ArgumentParser(description="Flocking Agents Simulation")

    # General settings
    parser.add_argument(
        "--mode", "-m", choices=["run", "plot"], help="Run or plot a simulation"
    )
    parser.add_argument(
        "--name", "-n", type=str, help="Unique name for this simulation"
    )
    parser.add_argument(
        "--debug_mode", action="store_true", help="Enable debug logging"
    )
    parser.add_argument("--dt", "-dt", type=float, help="Simulation time step")
    parser.add_argument("--wait", "-w", type=float, help="Wait time between rounds")

    # Agent settings
    parser.add_argument(
        "--max_speed",
        "-ms",
        type=float,
        help="maximum speed for agents (units per round)",
    )
    parser.add_argument(
        "--desired_distance", "-dd", type=float, help="desired distance between agents"
    )
    parser.add_argument(
        "--safe_distance", "-sd", type=float, help="safe distance between agents"
    )
    parser.add_argument(
        "--formation",
        "-form",
        type=str,
        help="formation of the agents (circle or square)",
    )
    parser.add_argument(
        "--agent_mode",
        "-am",
        type=str,
        choices=["influence", "naive", "basic", "plan"],
        help="agent mode: influence-based plan adoption, naive execution, basic (no planning), or plan (follow agent 0's plan)",
    )
    parser.add_argument(
        "--initial_position", "-ip", type=list, help="initial position of the agents"
    )
    parser.add_argument(
        "--randomize", "-rand", type=bool, help="randomize agent positions"
    )
    parser.add_argument(
        "--comm_range", "-cr", type=float, help="communication range of the agents"
    )

    # Simulation settings
    parser.add_argument("--seed", "-s", type=int, help="Random seed")
    parser.add_argument(
        "--spawn_x_min",
        "-sxmi",
        type=float,
        help="Lower bound of agent spawn x-coordinate",
    )
    parser.add_argument(
        "--spawn_x_max",
        "-sxma",
        type=float,
        help="Upper bound of agent spawn x-coordinate",
    )
    parser.add_argument(
        "--spawn_y_min",
        "-symi",
        type=float,
        help="Lower bound of agent spawn y-coordinate",
    )
    parser.add_argument(
        "--spawn_y_max",
        "-syma",
        type=float,
        help="Upper bound of agent spawn y-coordinate",
    )

    # Target settings
    parser.add_argument("--targets", "-t", type=int, help="Number of targets")
    parser.add_argument("--target_active", "-ta", type=bool, help="Activate targets")
    parser.add_argument("--target_speed", "-ts", type=list, help="Target speed")
    parser.add_argument("--target_position", "-tp", type=list, help="Target position")

    parser.add_argument("--rounds", "-r", type=int, help="Number of simulation rounds")
    parser.add_argument("--zombies", "-z", type=int, help="Number of inactive agents")
    parser.add_argument(
        "--agents", "-a", type=int, help="Number of agents in the simulation"
    )
    parser.add_argument(
        "--memory_limit", "-mlim", type=int, help="Agent message history memory limit"
    )
    parser.add_argument(
        "--collision_reprompt_attempts",
        "-cra",
        type=int,
        help="Maximum collision-driven reprompt attempts per round",
    )
    parser.add_argument(
        "--agent_response_timeout",
        "-art",
        type=float,
        help="Timeout (seconds) for each agent movement response",
    )
    parser.add_argument(
        "--collision_buffer",
        "-cb",
        type=float,
        help="Additional safety buffer added to safe_distance for collision checks",
    )
    parser.add_argument("--model", "-gpt", type=str, help="LLM model name")
    parser.add_argument(
        "--model_company",
        "-mc",
        type=str,
        choices=[
            "openai",
            "claude",
            "deepseek",
            "deepseek_api",
            "qwen",
            "llama_api",
        ],
        help="LLM model company",
    )
    parser.add_argument(
        "--reasoning_effort", "-ra", type=str, help="GPT reasoning ability"
    )

    # Plotting settings
    parser.add_argument("--x_min", "-pxmi", type=int, help="Plot left-side bound")
    parser.add_argument("--x_max", "-pxma", type=int, help="Plot right-side bound")
    parser.add_argument("--x_ticks", "-pxmt", type=int, help="Plot x-axis tick spacing")
    parser.add_argument("--y_min", "-pymi", type=int, help="Plot lower bound")
    parser.add_argument("--y_max", "-pyma", type=int, help="Plot upper bound")
    parser.add_argument("--y_ticks", "-pymt", type=int, help="Plot y-axis tick spacing")
    parser.add_argument(
        "--plot", "-plot", type=int, help="Enable plotting (1 for plot, 0 for not)"
    )

    parser.add_argument(
        "--leader_id", "-lid", type=int, help="Identifier of the leader agent"
    )

    parser.add_argument(
        "--use_pydantic_ai",
        action="store_true",
        help="Use Pydantic AI framework for Claude models (provides better structured output)",
    )

    args = parser.parse_args()
    config = load_config()

    # Merge CLI args with config (CLI args take priority)
    for key, value in config.items():
        if getattr(args, key) is None:  # Override only if not set via CLI
            setattr(args, key, value)

    return args
