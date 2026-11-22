#!/usr/bin/env bash
# ==============================================================================
# Autonomous LEGO Sorter - Universal Installer for Ubuntu / Debian Linux
# Installs all required system libraries, creates Python virtual environment,
# and installs dependencies.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "      Autonomous LEGO Sorter - Setup & Installation       "
echo "=========================================================="

# 1. Check for root / sudo if packages need installation
if command -v apt-get >/dev/null 2>&1; then
    echo "[1/4] Checking system dependencies (OpenGL, OpenCV, Python)..."
    SYS_PKGS="python3 python3-venv python3-pip libgl1 libglib2.0-0 libgomp1"
    
    # Check if packages installed
    MISSING_PKGS=""
    for pkg in $SYS_PKGS; do
        if ! dpkg -s "$pkg" >/dev/null 2>&1; then
            MISSING_PKGS="$MISSING_PKGS $pkg"
        fi
    done

    if [ -n "$MISSING_PKGS" ]; then
        echo "Installing missing system packages:$MISSING_PKGS"
        if [ "$(id -u)" -eq 0 ]; then
            apt-get update && apt-get install -y $MISSING_PKGS
        elif command -v sudo >/dev/null 2>&1; then
            sudo apt-get update && sudo apt-get install -y $MISSING_PKGS
        else
            echo "Warning: Not root and sudo not available. Continuing anyway..."
        fi
    else
        echo "All system packages are already installed."
    fi
fi

# 2. Setup Python virtual environment
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/4] Creating Python virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "[2/4] Virtual environment already exists at $VENV_DIR."
fi

# 3. Upgrade pip and install Python requirements
echo "[3/4] Installing Python requirements..."
"$VENV_DIR/bin/python" -m ensurepip --default-pip >/dev/null 2>&1 || true
"$VENV_DIR/bin/python" -m pip install --upgrade pip

"$VENV_DIR/bin/pip" install \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.28.0" \
    "opencv-python-headless>=4.9.0" \
    "pyserial>=3.5" \
    "onnxruntime>=1.18.0" \
    "numpy>=1.24.0" \
    "pillow>=10.0.0" \
    "tqdm>=4.66.0"

# 4. Check permissions on serial devices (dialout group)
echo "[4/4] Checking serial port permissions for Arduino..."
if [ -n "$USER" ]; then
    if ! id -nG "$USER" | grep -qw "dialout"; then
        echo "Note: To access USB serial ports without sudo, your user should be in the 'dialout' group."
        echo "Run: sudo usermod -a -G dialout $USER (and log out/in)."
    fi
fi

chmod +x start_sorter.sh 2>/dev/null || true

echo "=========================================================="
echo " Installation Complete! "
echo " Run ./start_sorter.sh to launch the application."
echo "=========================================================="
