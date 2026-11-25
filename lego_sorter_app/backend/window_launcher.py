#!/usr/bin/env python3
"""
Native App Window Launcher for LEGO Sorter
- Automatically detects desktop environment (Wayland / X11)
- Finds available browsers capable of standalone application window mode (--app=...)
  (Google Chrome, Chromium, Brave, Microsoft Edge, Flatpak Edge/Chrome, or Firefox Standalone Profile)
- Launches the GUI in its own dedicated, chromeless window (no URL bar, no tabs)
- Monitors the window process; when the window is closed, cleanly shuts down the backend server
"""

import os
import sys
import time
import shutil
import urllib.request
import logging
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

logger = logging.getLogger("lego_sorter.window")


def is_desktop_available() -> bool:
    """Checks whether a graphical desktop display (X11 or Wayland) is accessible."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def wait_for_server(url: str, timeout: float = 10.0) -> bool:
    """Polls server URL until it responds with HTTP 200 or timeout expires."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LegoSorterLauncher/1.0"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def find_browser_engine() -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Detects the best available engine to open a native-looking standalone window.
    Returns: (engine_type, command_template)
    """
    # 1. Native Chromium-based browsers (support --app=URL natively)
    native_chromiums = [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "brave-browser",
        "microsoft-edge",
        "microsoft-edge-stable"
    ]
    for b in native_chromiums:
        path = shutil.which(b)
        if path:
            return ("native_chromium", [path])

    # 2. Flatpak-installed Chromium-based browsers
    flatpaks = [
        "com.microsoft.Edge",
        "com.google.Chrome",
        "org.chromium.Chromium",
        "com.brave.Browser"
    ]
    if shutil.which("flatpak"):
        for fp in flatpaks:
            try:
                res = subprocess.run(
                    ["flatpak", "info", fp],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                if res.returncode == 0:
                    return ("flatpak_chromium", ["flatpak", "run", fp])
            except Exception:
                pass

    # 3. Firefox (can run with a dedicated chromeless profile)
    firefox_path = shutil.which("firefox")
    if firefox_path:
        return ("firefox", [firefox_path])

    # 4. Fallback: default xdg-open
    xdg_path = shutil.which("xdg-open")
    if xdg_path:
        return ("xdg_open", [xdg_path])

    return (None, None)


def _setup_firefox_app_profile() -> Path:
    """Configures a temporary Firefox profile with hidden tab bar and navigation bar."""
    profile_dir = Path(tempfile.gettempdir()) / "lego_sorter_firefox_app_profile"
    chrome_dir = profile_dir / "chrome"
    chrome_dir.mkdir(parents=True, exist_ok=True)

    user_js = profile_dir / "user.js"
    user_js.write_text("""
user_pref("toolkit.legacyUserProfileCustomizations.stylesheets", true);
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.tabs.warnOnClose", false);
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("browser.startup.page", 0);
""")

    user_css = chrome_dir / "userChrome.css"
    user_css.write_text("""
#TabsToolbar { visibility: collapse !important; }
#nav-bar { visibility: collapse !important; }
#PersonalToolbar { visibility: collapse !important; }
""")

    return profile_dir


def launch_native_window(
    port: int = 8000,
    host: str = "127.0.0.1",
    auto_shutdown_on_close: bool = True
) -> Optional[subprocess.Popen]:
    """
    Waits for the backend server and launches the GUI in its own native app window.
    """
    if not is_desktop_available():
        logger.info("Headless environment detected (no DISPLAY or WAYLAND_DISPLAY). Window launch skipped.")
        return None

    target_url = f"http://127.0.0.1:{port}"
    status_url = f"{target_url}/api/status"

    logger.info("Waiting for server to initialize before opening window...")
    if not wait_for_server(status_url, timeout=12.0):
        logger.warning(f"Server did not respond at {status_url} within 12 seconds. Aborting window launch.")
        return None

    engine_type, cmd_prefix = find_browser_engine()
    if not engine_type or not cmd_prefix:
        logger.warning("No suitable browser found to launch native window.")
        return None

    logger.info(f"Launching standalone GUI window using engine: {engine_type} ({' '.join(cmd_prefix)})")

    cmd: List[str] = []
    if engine_type in ("native_chromium", "flatpak_chromium"):
        cmd = cmd_prefix + [
            f"--app={target_url}",
            "--window-size=1440,920",
            "--window-position=center",
            "--class=lego-sorter-gui",
            "--name=LegoSorter"
        ]
    elif engine_type == "firefox":
        profile_dir = _setup_firefox_app_profile()
        cmd = cmd_prefix + [
            "--profile", str(profile_dir),
            "--new-window", target_url,
            "--class", "lego-sorter-gui"
        ]
    elif engine_type == "xdg_open":
        cmd = cmd_prefix + [target_url]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        logger.info(f"Native GUI window launched successfully (PID: {proc.pid}).")

        # Monitor thread: if the user closes the window, trigger graceful server shutdown
        if auto_shutdown_on_close and engine_type != "xdg_open":
            def _monitor_window():
                proc.wait()
                logger.info("Native GUI window was closed by the user.")
                try:
                    shutdown_url = f"http://127.0.0.1:{port}/api/control/shutdown"
                    req = urllib.request.Request(
                        shutdown_url,
                        data=b"{}",
                        headers={"Content-Type": "application/json", "User-Agent": "LegoSorterWindow/1.0"}
                    )
                    with urllib.request.urlopen(req, timeout=2.0) as resp:
                        pass
                except Exception:
                    pass

            monitor_thread = threading.Thread(target=_monitor_window, daemon=True)
            monitor_thread.start()

        return proc
    except Exception as e:
        logger.error(f"Failed to launch native GUI window: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    proc = launch_native_window(port=port)
    if proc:
        print(f"Window launched with PID {proc.pid}. Press Ctrl+C to exit.")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
