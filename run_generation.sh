#!/usr/bin/env bash
# Runner script for LEGO synthetic dataset generation
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    /home/ethan/.local/bin/uv venv .venv
    /home/ethan/.local/bin/uv pip install --python .venv numpy pillow tqdm
fi

echo "Running LEGO dataset generator..."
"$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/generate_dataset.py" "$@"
