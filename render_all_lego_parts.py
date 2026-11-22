#!/usr/bin/env bash
# Quick launcher to render every possible LEGO part in the LDraw library
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    /home/ethan/.local/bin/uv venv .venv
    /home/ethan/.local/bin/uv pip install --python .venv numpy pillow tqdm
fi

echo "Starting generation across ALL 23,800+ LEGO parts..."
echo "Backgrounds change dynamically on every frame to prevent color memorization."
echo "Use Ctrl+C anytime; progress is automatically saved and will resume from where you stopped."

"$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/generate_dataset.py" --all-parts --num-images 2000 "$@"
