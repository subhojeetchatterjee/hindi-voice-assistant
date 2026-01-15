#!/bin/bash

# ============================================================
# 🇮🇳 Offline Hindi Voice Assistant Runner
# ============================================================

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if venv exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first."
    exit 1
fi

# Run the assistant using the venv's python interpreter
echo "🚀 Starting Hindi Voice Assistant inside venv..."
"$SCRIPT_DIR/venv/bin/python3" "$SCRIPT_DIR/voice_assistant.py"
