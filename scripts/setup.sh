#!/bin/bash

# Setup script for RAG Agent System

set -e

echo "=========================================="
echo "RAG Agent System Setup"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from env.example..."
    cp env.example .env
    echo "⚠️  Please edit .env and add your OPENAI_API_KEY"
    echo ""
    read -p "Press Enter after you've added your OpenAI API key to .env..."
fi

# Check if OPENAI_API_KEY is set
source .env
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your-openai-api-key-here" ]; then
    echo "❌ Error: OPENAI_API_KEY is not set in .env file"
    echo "Please edit .env and add your actual OpenAI API key"
    exit 1
fi

# Install dependencies
echo "Installing dependencies with uv..."
uv sync
echo "✓ Dependencies installed"
echo ""

# Process data
echo "Processing data..."
uv run python -m src.preprocessing.process_data
echo "✓ Data processing completed"
echo ""

# Generate embeddings
echo "Generating embeddings (this may take a few minutes)..."
uv run python -m src.preprocessing.generate_embeddings
echo "✓ Embedding generation completed"
echo ""

echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run 'docker-compose up --build' to start the server"
echo "2. In another terminal, run 'uv run python -m src.evaluation.evaluate --dataset dev' to evaluate"
echo ""

