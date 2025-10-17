"""API schemas for RAG agent."""

from typing import Literal

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """Request schema for the RAG agent."""

    query: str = Field(
        ...,
        description="The multiple-choice legal question to be analyzed and answered",
    )


class AgentResponse(BaseModel):
    """Response schema for the RAG agent."""

    answer: Literal["A", "B", "C", "D"] = Field(
        ..., description="The letter of the correct option"
    )
