#!/usr/bin/env bash
# ==============================================================================
# Autonomous LEGO Sorter - Launcher Script
# Starts FastAPI server on 0.0.0.0:<PORT> and prints local network URLs
# Supports --port <PORT> or -p <PORT> flag
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default options
PORT=8000
EXTRA_ARGS=()

# Parse CLI arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --no-window|--server-only)
            EXTRA_ARGS+=("--no-window")
            shift
            ;;
        --help|-h)
            echo "Usage: ./start_sorter.sh [--port <PORT>] [--no-window]"
            echo "  --port, -p <PORT>       Specify the HTTP server port (default: 8000)"
            echo "  --no-window             Run in headless / server-only mode (do not open native GUI window)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Check virtualenv: prefer local app venv, or parent venv
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "$SCRIPT_DIR/../.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/../.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "=========================================================="
echo "          Starting Autonomous LEGO Sorter Server          "
echo "=========================================================="

# Determine local IP addresses
echo "Discovering local network IP addresses..."
IP_LIST=$(hostname -I 2>/dev/null || ip addr show | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)

echo ""
echo "Access the GUI from any browser on your network:"
echo "  - Localhost (This PC):    http://localhost:${PORT}"
for ip in $IP_LIST; do
    if [[ ! "$ip" =~ ^127\. ]] && [[ ! "$ip" =~ ^172\.17\. ]]; then
        echo "  - Phone / Laptop on LAN:  http://${ip}:${PORT}"
    fi
done
echo ""
echo "Native GUI window will launch automatically in its own window."
echo "Press Ctrl+C or close the window to stop the server."
echo "=========================================================="
echo ""

exec "$PYTHON_BIN" -m backend.main --port "$PORT" --host 0.0.0.0 "${EXTRA_ARGS[@]}"
