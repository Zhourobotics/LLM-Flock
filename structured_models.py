"""
Structured Models Module

This module defines Pydantic models for structured LLM output,
replacing regex-based parsing with robust JSON validation.
"""

from typing import List
from pydantic import BaseModel, Field, validator


class Position(BaseModel):
    """Represents a 2D coordinate position."""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    
    @validator('x', 'y')
    def validate_coordinates(cls, v):
        """Ensure coordinates are reasonable values."""
        if not isinstance(v, (int, float)):
            raise ValueError('Coordinates must be numeric')
        if abs(v) > 1000:  # Reasonable bounds for simulation
            raise ValueError('Coordinates must be within reasonable bounds (-1000 to 1000)')
        return float(v)


class AgentMoveResponse(BaseModel):
    """Response structure for agent movement decisions."""
    position: Position = Field(..., description="Target position for the agent")
    reasoning: str = Field(..., description="Explanation of why this position was chosen")

    
    @validator('reasoning')
    def validate_reasoning(cls, v):
        """Ensure reasoning is provided and not empty."""
        if not v or not v.strip():
            raise ValueError('Reasoning must be provided')
        return v.strip()


class PlanGenerationResponse(BaseModel):
    """Response structure for formation plan generation."""
    plan: List[Position] = Field(..., description="List of positions for the formation")
    my_plan_index: int = Field(..., description="Index of the position this agent will take")
    reasoning: str = Field(..., description="Explanation of the formation plan")
    
    @validator('plan')
    def validate_plan(cls, v):
        """Ensure plan has at least one position."""
        if not v:
            raise ValueError('Plan must contain at least one position')
        return v
    
    @validator('my_plan_index')
    def validate_plan_index(cls, v, values):
        """Ensure plan index is valid for the plan."""
        if 'plan' in values and values['plan']:
            if v < 0 or v >= len(values['plan']):
                raise ValueError(f'Plan index {v} is out of range for plan with {len(values["plan"])} positions')
        return v
    
    @validator('reasoning')
    def validate_reasoning(cls, v):
        """Ensure reasoning is provided and not empty."""
        if not v or not v.strip():
            raise ValueError('Reasoning must be provided')
        return v.strip()


# Utility functions for backward compatibility
def position_to_list(position: Position) -> List[float]:
    """Convert Position object to [x, y] list for legacy code."""
    return [position.x, position.y]


def plan_to_list(plan: List[Position]) -> List[List[float]]:
    """Convert plan positions to list of [x, y] coordinates for legacy code."""
    return [[pos.x, pos.y] for pos in plan]


def list_to_position(coords: List[float]) -> Position:
    """Convert [x, y] list to Position object."""
    if len(coords) != 2:
        raise ValueError("Coordinates must be a list of exactly 2 values")
    return Position(x=coords[0], y=coords[1])


def list_to_plan(coords_list: List[List[float]]) -> List[Position]:
    """Convert list of [x, y] coordinates to list of Position objects."""
    return [list_to_position(coords) for coords in coords_list]