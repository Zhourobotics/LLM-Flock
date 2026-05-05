"""
Structured Parser Module

This module provides parsing functionality for structured LLM output using
Pydantic models.
"""

import json
import logging
from typing import Optional, List, Tuple

from pydantic import ValidationError
from structured_models import (
    AgentMoveResponse,
    PlanGenerationResponse,
    position_to_list,
    plan_to_list,
)

logger = logging.getLogger(__name__)


class StructuredParser:
    """Parser for structured JSON responses validated by Pydantic models."""

    def __init__(self):
        """
        Initialize the structured parser.
        """
        self.json_success_count = 0
        self.json_failure_count = 0
    
    def parse_agent_move(self, response: str) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        Parse agent movement response.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Tuple of (position as [x, y], reasoning) or (None, None) if parsing fails
        """
        try:
            json_str = self._extract_json(response)
            if json_str:
                data = json.loads(json_str)
                move_response = AgentMoveResponse(**data)

                self.json_success_count += 1
                logger.debug(f"Successfully parsed agent move with JSON: {move_response.position}")

                return position_to_list(move_response.position), move_response.reasoning
        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            logger.debug(f"JSON parsing failed for agent move: {e}")
            self.json_failure_count += 1

        logger.warning(f"Failed to parse agent move response: {response[:100]}...")
        return None, None
    
    def parse_plan_generation(self, response: str) -> Tuple[Optional[List[List[float]]], Optional[int], Optional[str]]:
        """
        Parse plan generation response.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Tuple of (plan as [[x, y], ...], my_plan_index, reasoning) or (None, None, None) if parsing fails
        """
        try:
            json_str = self._extract_json(response)
            if json_str:
                data = json.loads(json_str)
                plan_response = PlanGenerationResponse(**data)

                self.json_success_count += 1
                logger.debug(f"Successfully parsed plan generation with JSON: {len(plan_response.plan)} positions")

                return plan_to_list(plan_response.plan), plan_response.my_plan_index, plan_response.reasoning
        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            logger.debug(f"JSON parsing failed for plan generation: {e}")
            self.json_failure_count += 1

        logger.warning(f"Failed to parse plan generation response: {response[:100]}...")
        return None, None, None
    
    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON object from text that might contain other content.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            JSON string if found, None otherwise
        """
        # Look for JSON object boundaries
        start = text.find('{')
        if start == -1:
            return None
        
        # Find the matching closing brace
        brace_count = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i+1]
        
        return None
    
    def get_parsing_stats(self) -> dict:
        """Get statistics about parsing success rates."""
        total = self.json_success_count + self.json_failure_count
        return {
            "total_parses": total,
            "json_success": self.json_success_count,
            "json_failed": self.json_failure_count,
            "json_success_rate": self.json_success_count / total if total > 0 else 0.0
        }

    def reset_stats(self):
        """Reset parsing statistics."""
        self.json_success_count = 0
        self.json_failure_count = 0


# Global parser instance
structured_parser = StructuredParser()


# Convenience functions that match the old API
def parse_agent_response(response: str) -> Tuple[Optional[List[float]], Optional[str]]:
    """Parse agent movement response. Returns (position, reasoning)."""
    return structured_parser.parse_agent_move(response)


def parse_plan_response(response: str) -> Tuple[Optional[List[List[float]]], Optional[int], Optional[str]]:
    """Parse plan generation response. Returns (plan, my_plan_index, reasoning)."""
    return structured_parser.parse_plan_generation(response)


# Legacy compatibility functions (for gradual migration)
def extract_position_structured(response: str) -> Optional[List[float]]:
    """Extract position with structured parsing."""
    position, _ = structured_parser.parse_agent_move(response)
    return position


def extract_plans_structured(response: str) -> Optional[List[List[float]]]:
    """Extract plan with structured parsing."""
    plan, _, _ = structured_parser.parse_plan_generation(response)
    return plan


def extract_my_plan_structured(response: str) -> Optional[int]:
    """Extract my_plan index with structured parsing."""
    _, my_plan, _ = structured_parser.parse_plan_generation(response)
    return my_plan
