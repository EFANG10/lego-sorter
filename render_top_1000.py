#!/usr/bin/env bash
# High-speed parallel renderer for the TOP 1,000 most common LEGO parts
# (Ranked by total occurrence across all official LEGO sets in history)
# Renders 2,000 images per part with dynamic multi-color backgrounds across all 16 CPU threads.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Setting up virtual environment..."
    /home/ethan/.local/bin/uv venv .venv
    /home/ethan/.local/bin/uv pip install --python .venv numpy pillow tqdm
fi

echo "=================================================================="
echo " Starting Top 1,000 LEGO Parts Synthetic Dataset Generator"
echo " Target: 1,000 parts x 2,000 images = 2,000,000 images (~19.7 GB)"
echo " Dynamic multi-color backgrounds active (anti-memorization)"
echo " Utilizing all 16 CPU threads"
echo " Safe to stop with Ctrl+C; automatically resumes anytime"
echo "=================================================================="

"$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/generate_dataset.py" --top-1000 --num-images 2000 "$@"
