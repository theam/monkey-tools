#!/bin/bash -l
# Wrapper script for Claude Desktop/Code MCP server startup.
# Uses login shell (-l) to inherit user's PATH (needed for GUI apps to find uv).
export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/transcription-mcp/.venv"
cd "$(dirname "$0")"
exec uv run python server.py
