import pickle
import os
import logging

from datetime import datetime

logger = logging.getLogger(__name__)


class Data:
    """
    A class to facilitate storage and retrieval of test data
    """

    def __init__(self, identifier, agents, settings, results):
        self.identifier = identifier
        self.directory = f"./{results}/{identifier}"
        self.agents = agents
        self.settings = settings
        self.results = results

    @staticmethod
    def save(
        agents,
        settings,
        identifier,
        results="results",
        last_round=None,  # new optional parameter for last completed round
    ) -> str:
        Data.check_for_results_directory(
            results
        )  # make sure we have a results directory
        dir_path = f"./{results}/{identifier}"

        # If last_round is provided, trim each agent's history
        agents_object = [
            {
                "identifier": agent.identifier,
                "position_history": (
                    agent.position_history
                    if last_round is None
                    else agent.position_history[:last_round]
                ),
                "memory": agent.memory,
                "chat_history": agent.chat_history,
                "model": agent.model,
                "model_company": agent.model_company,
                "plan": agent.plan,
                "my_plan": agent.my_plan,
                "my_goal": agent.my_goal,
                # Ensure plan source tracking is saved
                "plan_source_id": getattr(agent, "plan_source_id", agent.identifier),
                "plan_origin_id": getattr(agent, "plan_origin_id", agent.identifier),
                "parent_agent": (
                    agent.parent_agent.identifier if agent.parent_agent else None
                ),
                "plan_origin": agent.plan_origin,
            }
            for agent in agents
        ]

        for i, agent_obj in enumerate(agents_object):
            logger.debug(
                "Saving Agent %d with plan_source_id=%s plan_origin_id=%s",
                i,
                agent_obj["plan_source_id"],
                agent_obj["plan_origin_id"],
            )

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)  # create an empty test results directory

        # Save the data to file (overwriting any partial runs)
        with open(f"{dir_path}/results", "wb") as file:
            pickle.dump(Data(identifier, agents_object, settings, results), file)

        Data.save_dialog_history(agents_object, dir_path)
        logger.info(
            f"Success: Saved test {identifier} (up to round {last_round if last_round is not None else settings.rounds})! Dialog history saved to {dir_path}/dialog_history.txt"
        )
        return identifier

    @staticmethod
    def load(args, results="results"):
        file_path = f"./{results}/{args.name}/results"
        with open(file_path, "rb") as file:
            data = pickle.load(file)

        # Save dialog history separately when loading
        dir_path = f"./{results}/{args.name}"
        Data.save_dialog_history(data.agents, dir_path)

        # Load plot settings in :)
        data.settings.x_min = args.x_min
        data.settings.x_max = args.x_max
        data.settings.x_ticks = args.x_ticks
        data.settings.y_min = args.y_min
        data.settings.y_max = args.y_max

        # Log full chat history for each agent (not truncated memory)
        for agent in data.agents:
            logger.info(f"\n=== Agent {agent['identifier']} Dialog History ===")
            for entry in agent["chat_history"]:
                # Skip malformed entries (safety check)
                if not isinstance(entry, dict) or "role" not in entry:
                    continue
                    
                role = entry["role"]
                
                # Handle different content formats across model companies
                if isinstance(entry["content"], list):
                    # For most models: content is wrapped in prompt_assemble format
                    content = entry["content"][0] if entry["content"] else ""
                    # Extract text from {"type": "text", "text": "message"} format
                    if isinstance(content, dict) and "text" in content:
                        content = content["text"]
                else:
                    # For deepseek_api/llama_api: content is direct string
                    content = entry["content"]
                logger.info(f"[{role.upper()}]: {content}")

        logger.info(f"\nSuccess: Loaded test {args.name}!")

        return data

    @staticmethod
    def check_for_results_directory(dirname):
        # Create our results directory
        if not os.path.exists(f"./{dirname}"):
            os.makedirs(f"./{dirname}")

    @staticmethod
    def save_dialog_history(agents, dir_path):
        """Saves agent dialog history to a text file."""
        dialog_file = f"{dir_path}/dialog_history.txt"
        with open(dialog_file, "w", encoding="utf-8") as log_file:
            for agent in agents:
                log_file.write(
                    f"\n=== Agent {agent['identifier']} Dialog History ===\n"
                )
                # Use chat_history (full conversation) instead of memory (truncated)
                for entry in agent["chat_history"]:
                    # Skip malformed entries (safety check)
                    if not isinstance(entry, dict) or "role" not in entry:
                        continue
                        
                    role = entry["role"]
                    
                    # Handle different content formats across model companies
                    if isinstance(entry["content"], list):
                        # For most models: content is wrapped in prompt_assemble format
                        content = entry["content"][0] if entry["content"] else ""
                        # Extract text from {"type": "text", "text": "message"} format
                        if isinstance(content, dict) and "text" in content:
                            content = content["text"]
                    else:
                        # For deepseek_api/llama_api: content is direct string
                        content = entry["content"]
                    
                    log_file.write(f"[{role.upper()}]: {content}\n")
