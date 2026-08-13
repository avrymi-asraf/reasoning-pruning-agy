#!/usr/bin/env bash
set -e

echo "=== Setting up Reasoning Pruning Environment for Google Colab ==="

# 1. Install uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Sync dependencies
echo "Syncing dependencies with uv..."
uv sync --extra dev

echo "=== Setup Complete! ==="
echo "You can now run commands with: uv run rp --help"
echo "Or start the interactive visualizer with: uv run rp view --share"
