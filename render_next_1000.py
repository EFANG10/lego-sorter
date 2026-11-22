#!/usr/bin/env bash
# High-speed parallel renderer for the NEXT 1,000 most common LEGO parts (ranks 1,001 - 2,000)
# Includes 237 Technic gears, axles, connectors, beams, and wheels.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Setting up virtual environment..."
    /home/ethan/.local/bin/uv venv .venv
    /home/ethan/.local/bin/uv pip install --python .venv numpy pillow tqdm
fi

echo "=================================================================="
echo " Starting Next 1,000 LEGO Parts Synthetic Dataset Generator"
echo " (Ranks 1,001 - 2,000: Technic, gears, connectors, wheels, wedges)"
echo " Dynamic multi-color backgrounds active"
echo " Utilizing all CPU threads via headless OSMesa"
echo " Safe to stop with Ctrl+C; automatically resumes anytime"
echo "=================================================================="

"$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/generate_dataset.py" --parts-file "$SCRIPT_DIR/next_1000_parts.txt" --num-images 2000 "$@"
