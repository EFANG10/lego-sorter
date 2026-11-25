# Autonomous LEGO Sorter - Web GUI & Hardware Controller

An end-to-end, high-performance web dashboard, OpenCV computer vision pipeline, and Arduino gate controller for automated LEGO brick sorting.

Designed to run on Ubuntu (24.04 LTS & derivatives) as a standalone application accessible from any web browser on your local network (laptop, desktop, or mobile phone).

---

## Key Features

1. **Physical Bin Mapping & Two-Sided Conveyor Schematic**:
   - Assign LEGO parts to physical bins `0` through `N-1`.
   - **Dynamic Bins**: Add new bins or delete existing bins with permanent automatic persistence across restarts.
   - **Two-Sided Chute Layout (Top & Bottom)**: Add or remove conveyor spots dynamically with chutes on both sides (Top/North diverters and Bottom/South diverters) plus vertical tiered rows.
   - **Multiple parts per bin**: Any number of LEGO parts can be mapped into the same bin.
   - **Bin -1 (End-of-Belt Overflow / Reject)**: Unassigned parts or bricks with low classifier confidence automatically pass through to the end of the belt into Bin -1.
   - **Presets Toolbar**: Save, load, and delete named sorting presets (e.g., "Color Sorter", "Technic Sorting", "Basic Bricks").
   - **Clear & Remove Actions**: Clear an individual bin's parts, clear all bins, or remove all bins with modal confirmation guards.
   - **Drag & Drop Reordering**: Drag and drop bin nodes between conveyor spots, sides (Top/Bottom), and vertical rows.
   - **Direct Part Drag & Drop**: Drag parts from the Unassigned Bricks panel directly onto schematic chute nodes or bin cards.

2. **Per-Bin Hardware Delay & Pulse Controls**:
   - Each bin card features independent travel delay settings:
     - **Belt Delay (`delay_ms`)**: Fine-tune the exact physical travel time (in ms) from camera capture to that specific master gate.
     - **Gate Pulse (`pulse_ms`)**: How long the deflection gate stays open.

3. **1,100+ Unassigned Bricks Panel & Categorized Part Picker Modal**:
   - **Unassigned Bricks Panel**: Browse all 1,100+ unassigned parts with live search, category dropdown, and pagination. Drag any brick directly into a bin!
   - **"+ Assign Parts" Modal**: Multi-select modal with prominent category pills (`All`, `Bricks`, `Plates`, `Slopes`, `Tiles`, `Technic`, `Minifigures`, etc.) and canonical 3D LDraw isometric thumbnails (neutral 30° lat, 45° lon CAD viewing angle).

4. **Direct Physical Camera Input, In-Frame Capture & Multi-Snapshot Inspection**:
   - Captures directly from physical USB webcams (`/dev/video0`, `/dev/video1`, etc.) with clear standby screen if disconnected.
   - **No Static Trigger Gate**: Objects are detected using OpenCV contours and classified once the brick is fully within camera frame margins.
   - **Actual Captured Snapshot Series**: Displays the exact series of ~5 camera snapshots captured as the brick passed by, shown directly in the UI alongside predicted part, category, and top-5 averaged confidences.
   - Multi-snapshot inference eliminates glare, shadows, and motion blur.

5. **Arduino Gate Actuation (USB Serial)**:
   - Dispatches `BIN:<id>:<pulse>\n` over USB Serial to Arduino at 115200 baud.
   - Non-blocking hardware gate timing and automatic simulation fallback when hardware is not plugged in.
   - Arduino sketch provided in [`arduino/lego_sorter_gates.ino`](arduino/lego_sorter_gates.ino).

6. **Native Standalone Window & PWA Support**:
   - The GUI automatically launches in its **own dedicated window** (no browser URL bar, no tabs, no search box) via native Chromium application mode (`--app`) or isolated profile.
   - Looks and behaves like a native desktop app with full window controls and taskbar icon.
   - Closing the native window cleanly shuts down the server, releases camera and serial resources, and unbinds the port.
   - Desktop entry installed to `~/.local/share/applications/lego-sorter.desktop` for 1-click launch from the Ubuntu application menu.
   - PWA manifest included for standalone app mode on mobile phones and tablets.

7. **Custom Port Selection & Clean Shutdown**:
   - Start the server on any custom network port:
     ```bash
     ./start_sorter.sh --port 8080
     ```
   - Run in server-only / headless mode without opening a desktop window:
     ```bash
     ./start_sorter.sh --no-window
     ```
   - **⏻ Shutdown Button**: Stops camera tracking, closes serial ports, and cleanly unbinds the port from the browser UI or terminal.

---

## Quickstart

### 1. Installation
Run the installer script:
```bash
cd lego_sorter_app
./install.sh
```

### 2. Launch the Application
Start the server (default port 8000, or choose your own):
```bash
./start_sorter.sh
# or specify port:
./start_sorter.sh --port 8080
```

The launcher will display your local network URLs:
- **Localhost**: `http://localhost:8000`
- **Phone / Laptop on LAN**: `http://192.168.x.x:8000`

---

## Arduino Hardware Setup
1. Open `arduino/lego_sorter_gates.ino` in the Arduino IDE.
2. Select your board (Uno, Nano, Mega) and USB port.
3. Wire your servo motors or solenoid relay drivers to digital pins `D2` through `D7`.
4. Click **Upload**.
5. In the Web UI **Settings** tab, choose your serial port from the dropdown.
