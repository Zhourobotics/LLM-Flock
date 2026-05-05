"""
API Client Module

This module handles all API interactions with Language Models, including
rate limiting, retry logic, and performance monitoring.
"""

import asyncio
import time
import csv
import logging
from datetime import datetime

from response_parser import clean_llm_response_text
from structured_parser import parse_agent_response, parse_plan_response
from structured_models import AgentMoveResponse, PlanGenerationResponse

logger = logging.getLogger(__name__)

# Global semaphore for rate limiting
SEMAPHORE = asyncio.Semaphore(10)  # Limit concurrent API requests


async def call_agent(agent, message, model, memory_limit, response_format=None):
    """Wrap an agent's API call with a semaphore to limit concurrent requests."""
    async with SEMAPHORE:  # Limit concurrent API requests
        return await retry_api_call(agent, message, model, memory_limit, retries=3, response_format=response_format)


async def retry_api_call(agent, message, model, memory_limit, retries=3, response_format=None):
    """Retries API calls if the response format is incorrect or if the API fails."""
    for attempt in range(retries):
        start_time = time.time()

        try:
            response = await agent.prompt(message, model, memory_limit, response_format)

            # Clean the response text first
            cleaned_response = clean_llm_response_text(response)

            parsed_ok = False
            parsed_position = None

            if response_format == AgentMoveResponse:
                parsed_position, _ = parse_agent_response(cleaned_response)
                parsed_ok = parsed_position is not None
            elif response_format == PlanGenerationResponse:
                plan_data, my_plan_idx, _ = parse_plan_response(cleaned_response)
                parsed_ok = plan_data is not None and my_plan_idx is not None
            else:
                # Keep default behavior permissive when no strict response model is requested.
                parsed_ok = True

            if parsed_ok:
                end_time = time.time()
                response_time = end_time - start_time
                logger.info(
                    f"Agent {agent.identifier} | Response Time: {response_time:.2f}s | Parsed Position: {parsed_position}"
                )
                await log_api_performance(agent.identifier, response_time, "Success")
                return response

            logger.warning(
                f"Agent {agent.identifier} | Structured parsing failed for response: {response}"
            )
            message += (
                "\nYour response must be valid JSON matching the requested schema.\n"
            )

        except Exception as e:
            wait_time = 2**attempt  # Exponential backoff
            logger.warning(f"Agent {agent.identifier} | Retry {attempt+1} | Error: {e}")
            await asyncio.sleep(wait_time)

    logger.error(f"Agent {agent.identifier} | All retries failed.")
    await log_api_performance(agent.identifier, None, "Failed")
    return None


async def log_api_performance(agent_id, response_time, status):
    """Save API performance metrics to a CSV file."""
    with open("api_performance.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([datetime.now(), agent_id, response_time, status])
