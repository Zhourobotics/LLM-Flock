"""
Pydantic AI Wrapper Module

This module provides a wrapper around Pydantic AI agents for use with Claude models,
offering better structured output handling compared to direct API calls.
"""

import logging
from typing import Optional

try:
    from pydantic_ai import Agent as PydanticAgent
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False

from structured_models import AgentMoveResponse, PlanGenerationResponse
from keys import get_key

logger = logging.getLogger(__name__)


def create_memory_limit_processor(memory_limit: int):
    """Create a history processor that respects memory_limit like the direct API approach."""
    async def processor(messages):
        # Follow same logic as direct API: keep first message + last memory_limit * 2 messages
        if len(messages) > memory_limit * 2 + 1:
            return [messages[0]] + messages[-memory_limit * 2:]
        return messages
    return processor


class PydanticAIWrapper:
    """Wrapper for Pydantic AI agents to provide structured output for Claude models."""

    def __init__(self, model_name: str, memory_limit: int = 5):
        """
        Initialize Pydantic AI wrapper for Claude.

        Args:
            model_name: Claude model name (e.g., 'claude-3-5-sonnet-20241022')
            memory_limit: Maximum number of message pairs to keep in history
        """
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError(
                "Pydantic AI is not available. Install with: pip install pydantic-ai"
            )

        # Initialize Anthropic model for Pydantic AI
        self.model = AnthropicModel(
            model_name=model_name, provider=AnthropicProvider(api_key=get_key("claude"))
        )

        # Create memory processor with the same logic as direct API approach
        self.memory_limit = memory_limit
        self.history_processor = create_memory_limit_processor(memory_limit)
        
        # Create specialized agents for different response types with memory management
        self.movement_agent = PydanticAgent(
            model=self.model,
            output_type=AgentMoveResponse,
            system_prompt="You are a drone navigating in a 2D space. Respond with structured JSON.",
            history_processors=[self.history_processor]
        )

        self.planning_agent = PydanticAgent(
            model=self.model,
            output_type=PlanGenerationResponse,
            system_prompt="You are a drone planning coordinator. Respond with structured JSON.",
            history_processors=[self.history_processor]
        )
        
        # Store conversation history for continuity
        self.movement_history = None
        self.planning_history = None

    async def call_movement_agent(
        self, system_prompt: str, user_message: str, use_history: bool = True
    ) -> AgentMoveResponse:
        """
        Call movement agent with structured output.

        Args:
            system_prompt: System instructions for the agent
            user_message: User query about movement

        Returns:
            Validated AgentMoveResponse object
        """
        try:
            # Update system prompt dynamically
            self.movement_agent = PydanticAgent(
                model=self.model,
                output_type=AgentMoveResponse,
                system_prompt=system_prompt,
                history_processors=[self.history_processor],
            )

            # Run the agent with message history for continuity
            result = await self.movement_agent.run(
                user_message, 
                message_history=self.movement_history if use_history else None
            )
            logger.debug(f"Pydantic AI movement result: {result.output}")
            
            # Update conversation history for next call
            if use_history:
                self.movement_history = result.new_messages()
            else:
                self.movement_history = None

            return result.output

        except Exception as e:
            logger.error(f"Pydantic AI movement call failed: {e}")
            raise

    async def call_planning_agent(
        self, system_prompt: str, user_message: str
    ) -> PlanGenerationResponse:
        """
        Call planning agent with structured output.

        Args:
            system_prompt: System instructions for the agent
            user_message: User query about planning

        Returns:
            Validated PlanGenerationResponse object
        """
        try:
            # Update system prompt dynamically
            self.planning_agent = PydanticAgent(
                model=self.model,
                output_type=PlanGenerationResponse,
                system_prompt=system_prompt,
                history_processors=[self.history_processor],
            )

            # Run the agent with message history for continuity
            result = await self.planning_agent.run(
                user_message,
                message_history=self.planning_history
            )
            logger.debug(f"Pydantic AI planning result: {result.output}")
            
            # Update conversation history for next call
            self.planning_history = result.new_messages()

            return result.output

        except Exception as e:
            logger.error(f"Pydantic AI planning call failed: {e}")
            raise


def create_pydantic_ai_wrapper(model_name: str) -> Optional[PydanticAIWrapper]:
    """
    Factory function to create Pydantic AI wrapper.

    Args:
        model_name: Claude model name

    Returns:
        PydanticAIWrapper instance or None if not available
    """
    if not PYDANTIC_AI_AVAILABLE:
        logger.warning("Pydantic AI not available, falling back to direct API")
        return None

    try:
        return PydanticAIWrapper(model_name)
    except Exception as e:
        logger.error(f"Failed to create Pydantic AI wrapper: {e}")
        return None
