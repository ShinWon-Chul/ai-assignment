"""FastAPI main application."""

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI

from src.agent.rag_agent import RAGAgent
from src.api.schemas import AgentRequest, AgentResponse

# Load environment variables
load_dotenv()

# Global agent instance
agent: RAGAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and cleanup resources."""
    global agent
    # Initialize agent
    agent = RAGAgent()
    yield
    # Cleanup if needed
    agent = None


app = FastAPI(
    title="RAG Agent for Legal Questions",
    description="RAG-based agent system for answering legal multiple-choice questions",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "RAG Agent API is running"}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict")
async def predict(request: AgentRequest) -> AgentResponse:
    """Predict the answer for a given legal question."""
    if agent is None:
        raise RuntimeError("Agent not initialized")

    answer = await agent.predict(request.query)
    return AgentResponse(answer=answer)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

