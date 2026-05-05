from openai import OpenAI
from anthropic import Anthropic
import asyncio
import logging


from prompts import Prompt, prompt_assemble
from arguments import parse_cli_arguments
from keys import get_key, get_base_url
from structured_models import (
    AgentMoveResponse,
    PlanGenerationResponse,
    position_to_list,
)


args = parse_cli_arguments()

logger = logging.getLogger(__name__)


class Agent:
    # identifier = ""
    # latest = ""
    # position = None
    # "developer" for the new o-series models

    # active = True
    # TODO: add modes for the agents

    def __init__(self, identifier, position, active=True):
        self.identifier = identifier
        self.position = position
        self.position_history = [position]
        self.chat_history = []

        self.model = args.model
        self.model_company = args.model_company
        # Initialize the client based on model type
        self.openai_api_models = [
            "openai",
            "deepseek_api",
            "llama_api",
            "qwen",
            "deepseek",
        ]

        # most of the models use OpenAI API style
        if self.model_company in self.openai_api_models:
            self.client = OpenAI(
                api_key=get_key(self.model_company),
                base_url=get_base_url(self.model_company),
            )
        elif self.model_company == "claude":
            self.client = Anthropic(api_key=get_key(self.model_company))
            # Each agent gets its own Pydantic AI wrapper for independent conversation history
            if args.use_pydantic_ai:
                from pydantic_ai_wrapper import PydanticAIWrapper

                self.pydantic_ai_wrapper = PydanticAIWrapper(
                    self.model, args.memory_limit
                )
            else:
                self.pydantic_ai_wrapper = None
        # For llama_api, client is initialized above with OpenAI compatibility

        self.active = active
        self.developer_msg = self.get_developer_msg()

        # Initialize memory based on model type
        if self.model_company == "deepseek_api" or self.model_company == "llama_api":
            self.memory = [{"role": "system", "content": self.developer_msg}]
        elif self.model_company == "claude":
            self.memory = []
        else:
            self.memory = [
                # {"role": "system", "content": [prompt_assemble(self.developer_msg)]}
                {"role": "system", "content": self.developer_msg}
            ]
        self.position_history = []

        # Initialize chat_history with the initial system message from memory
        self.chat_history.extend(self.memory)
        self.comm_range = args.comm_range
        self.last_id = None

        self.comm_neighbors = None
        self.parent_agent = None
        self.plan = None
        self.my_plan = None
        self.my_goal = None
        self.influence = None
        self.plan_subscriber = 0

        # Initialize plan source tracking - explicitly set to own ID
        self.plan_source_id = identifier  # Initially, each agent is its own plan source
        self.plan_origin_id = identifier  # Track the original source of the plan
        self.plan_origin = []

        print(
            f"Initialized Agent {self.identifier} with model {self.model} from {self.model_company} with api base {get_base_url(self.model_company)} and api key starting with {get_key(self.model_company)[:4]}..."
        )

    def get_developer_msg(
        self,
        formation=args.formation,
        target_position=args.target_position,
        desired_distance=args.desired_distance,
        safe_distance=args.safe_distance,
        max_speed=args.max_speed,
    ):
        return (
            Prompt.agent_role
            + Prompt.agent_requirement.format(
                formation,
                target_position,
                desired_distance,
                safe_distance,
                max_speed,
            )
            + Prompt.agent_task
        )

    async def prompt(self, message, model, memory_limit, response_format=None):
        is_movement_call = response_format == AgentMoveResponse
        completion = None

        if not self.active:  # zombie agent
            self.latest = "Reasoning: STATIC AGENT\nPosition: {}".format(self.position)
            self.memory.append(
                # {"role": "assistant", "content": [prompt_assemble(self.latest)]}
                {"role": "assistant", "content": self.latest}
            )
            self.chat_history.append(
                # {"role": "assistant", "content": [prompt_assemble(self.latest)]}
                {"role": "assistant", "content": self.latest}
            )
            return self.latest

        # Movement updates use only: first system message + current user prompt.
        if is_movement_call:
            user_message = {"role": "user", "content": message}
            if self.model_company == "claude":
                messages_for_api = [user_message]
            else:
                system_message = (
                    self.memory[0]
                    if self.memory and self.memory[0].get("role") == "system"
                    else {"role": "system", "content": self.developer_msg}
                )
                messages_for_api = [system_message, user_message]
            self.chat_history.append(user_message)
        else:
            # Planning and other calls keep the existing bounded conversation memory.
            memory_hist = self.memory
            if len(self.memory) > memory_limit * 2 + 1:
                self.memory = [memory_hist[0]] + memory_hist[-memory_limit * 2 :]

            if self.model_company == "deepseek_api" or self.model_company == "llama_api":
                self.memory.append({"role": "user", "content": message})
                self.chat_history.append({"role": "user", "content": message})
            else:
                self.memory.append(
                    # {"role": "user", "content": [prompt_assemble(message)]}
                    {"role": "user", "content": message}
                )

                self.chat_history.append(
                    # {"role": "user", "content": [prompt_assemble(message)]}
                    {"role": "user", "content": message}
                )

            messages_for_api = self.memory

        try:
            # API calling for OpenAI
            if self.model_company in self.openai_api_models:
                # Use Pydantic model for structured output if provided, else fallback to generic JSON
                if response_format:
                    format_param = response_format
                else:
                    format_param = {"type": "json_object"}

                # gpt-5 family uses the Responses API with structured reasoning;
                # everything else (including non-OpenAI providers reached via the
                # OpenAI-compatible client) uses chat.completions with JSON mode.
                if model.startswith("gpt-5"):
                    completion = await asyncio.to_thread(
                        lambda: self.client.responses.parse(
                            model=model,
                            input=messages_for_api,
                            reasoning={"effort": args.reasoning_effort},
                            text_format=format_param,
                        )
                    )
                    self.latest = completion.output_text
                else:
                    completion = await asyncio.to_thread(
                        lambda: self.client.chat.completions.create(
                            model=model,
                            messages=messages_for_api,
                            response_format={"type": "json_object"},
                        )
                    )
                    self.latest = completion.choices[0].message.content

            # API calling for Anthropic Claude
            elif self.model_company == "claude":
                # Use Pydantic AI if enabled and available, otherwise direct API
                if self.pydantic_ai_wrapper and response_format:
                    await self._call_claude_with_pydantic_ai(message, response_format)
                else:
                    # Direct API call (original approach)
                    completion = await asyncio.to_thread(
                        lambda: self.client.messages.create(
                            model=model,
                            messages=messages_for_api,
                            system=self.developer_msg,
                            max_tokens=2048,
                        )
                    )
                    self.latest = completion.content[0].text

        except Exception as e:
            print(f"Error calling model for agent {self.identifier}: {str(e)}")
            self.latest = f"Error: {str(e)}\nPosition: {self.position}"

        # Keep movement calls stateless in memory while preserving full chat logs.
        if is_movement_call:
            self.chat_history.append({"role": "assistant", "content": self.latest})
            if model.startswith("gpt-5") and completion is not None:
                self.last_id = completion.id
            return self.latest

        # Store the response in memory for non-movement calls.
        if self.model_company == "deepseek_api" or self.model_company == "llama_api":
            self.memory.append({"role": "assistant", "content": self.latest})
            self.chat_history.append({"role": "assistant", "content": self.latest})

        elif model.startswith("gpt-5"):
            self.chat_history.append({"role": "assistant", "content": self.latest})
            if completion is not None:
                self.last_id = completion.id

        else:
            self.memory.append({"role": "assistant", "content": self.latest})
            self.chat_history.append({"role": "assistant", "content": self.latest})

        return self.latest

    async def _call_claude_with_pydantic_ai(self, message, response_format):
        """Handle Claude API calls using Pydantic AI framework for structured output."""
        try:
            # Determine which agent type to use based on response format
            if response_format == AgentMoveResponse:
                # Call movement agent with full context
                result = await self.pydantic_ai_wrapper.call_movement_agent(
                    system_prompt=self.developer_msg,
                    user_message=message,
                    use_history=False,
                )
                # Convert structured response back to JSON string for compatibility
                self.latest = f'{{"position": {{"x": {result.position.x}, "y": {result.position.y}}}, "reasoning": "{result.reasoning}"}}'

            elif response_format == PlanGenerationResponse:
                # Call planning agent
                result = await self.pydantic_ai_wrapper.call_planning_agent(
                    system_prompt=self.developer_msg, user_message=message
                )
                # Convert structured response back to JSON string for compatibility
                plan_json = ", ".join(
                    [f'{{"x": {pos.x}, "y": {pos.y}}}' for pos in result.plan]
                )
                self.latest = f'{{"plan": [{plan_json}], "my_plan_index": {result.my_plan_index}, "reasoning": "{result.reasoning}"}}'

            else:
                # Fallback to direct API for unknown formats
                logger.warning(
                    f"Unknown response format {response_format}, falling back to direct API"
                )
                completion = await asyncio.to_thread(
                    lambda: self.client.messages.create(
                        model=self.model,
                        messages=self.memory,
                        system=self.developer_msg,
                        max_tokens=2048,
                    )
                )
                self.latest = completion.content[0].text

            logger.info(
                f"Agent {self.identifier} | Used Pydantic AI for Claude structured output"
            )

        except Exception as e:
            logger.error(f"Pydantic AI call failed for agent {self.identifier}: {e}")
            # Fallback to direct API call
            logger.info(f"Agent {self.identifier} | Falling back to direct Claude API")
            completion = await asyncio.to_thread(
                lambda: self.client.messages.create(
                    model=self.model,
                    messages=self.memory,
                    system=self.developer_msg,
                    max_tokens=2048,
                )
            )
            self.latest = completion.content[0].text

    def update(self, position):
        self.position = position
        self.position_history.append(self.position)

    # update the developer message on the air
    # takes args as defualt, but takes input as new values
    def update_developer_msg(
        self,
        formation=args.formation,
        target_position=args.target_position,
        desired_distance=args.desired_distance,
        safe_distance=args.safe_distance,
        max_speed=args.max_speed,
        plan=None,
    ):
        self.developer_msg = self.get_developer_msg(
            formation, target_position, desired_distance, safe_distance, max_speed
        )
        self.developer_msg += Prompt.plan_general.format(plan)
        self.memory[0]["content"] = [prompt_assemble(self.developer_msg)]

    def __str__(self):
        return "[{} Agent2D: (x: {}, y: {})]".format(
            self.identifier, self.position[0], self.position[1]
        )


class Target:
    def __init__(self, identifier, position, active=False, speed=[0, 0], dt=1):
        self.identifier = identifier
        self.position = position
        self.speed = speed
        self.active = active
        self.dt = dt

    def update(self):
        self.position += self.velocity * self.dt if self.active else [0, 0]
