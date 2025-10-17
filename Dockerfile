# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY src ./src
COPY data ./data

# Install dependencies
RUN uv sync --no-dev

# Copy vector store (must be created before Docker build)
COPY vector_store ./vector_store

# Expose port
EXPOSE 8000

# Set environment variable
ENV PORT=8000

# Run the FastAPI server
CMD ["uv", "run", "python", "-m", "src.api.main"]

