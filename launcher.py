#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of OCR Butterfly.
# Based on MLX-Video-OCR-DeepSeek-Apple-Silicon by matica0902 (AGPL-3.0).
# Copyright (C) 2025 MLX DeepSeek-OCR contributors
# Copyright (C) 2026 Ricardo Kupper
# See the LICENSE file in the project root for full license text.

"""
OCR Butterfly — Native App Launcher
====================================
Shows a loading window IMMEDIATELY (so macOS sees a GUI app right away),
starts Flask in the background, then navigates to the app once ready.
Closing the window kills the Flask server.
"""

import os
import sys
import time
import signal
import socket
import threading
import subprocess
import atexit
import logging
from importlib.metadata import PackageNotFoundError, version

# ---------------------------------------------------------------------------
# Logging — writes to file so we can debug .app issues
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

LOG_PATH = os.path.join(PROJECT_DIR, "launcher.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("launcher")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_TITLE = "OCR Butterfly"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 900
MIN_WIDTH = 900
MIN_HEIGHT = 600
HOST = "127.0.0.1"
PORT_RANGE = range(5001, 5011)

# ---------------------------------------------------------------------------
# Loading screen HTML (shown instantly while Flask boots)
# ---------------------------------------------------------------------------
LOADING_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", sans-serif;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    color: #e2e8f0;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100vh;
    overflow: hidden;
  }
  .container {
    text-align: center;
    animation: fadeIn 0.5s ease-out;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .icon {
    font-size: 64px;
    margin-bottom: 24px;
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1); }
  }
  h1 {
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 12px;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .status {
    font-size: 16px;
    color: #94a3b8;
    margin-bottom: 32px;
  }
  .spinner {
    width: 48px;
    height: 48px;
    border: 3px solid rgba(96, 165, 250, 0.2);
    border-top-color: #60a5fa;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 24px;
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  .steps {
    font-size: 13px;
    color: #64748b;
    line-height: 1.8;
  }
  .steps .done { color: #4ade80; }
  .steps .active { color: #60a5fa; }
  #step-server, #step-model, #step-ready { transition: color 0.3s; }
</style>
</head>
<body>
<div class="container">
  <div class="icon">🦋</div>
  <h1>OCR Butterfly</h1>
  <p class="status" id="statusText">Starting up...</p>
  <div class="spinner" id="spinner"></div>
  <div class="steps">
    <div id="step-server" class="active">● Starting server...</div>
    <div id="step-model">○ Loading OCR model...</div>
    <div id="step-ready">○ Ready</div>
  </div>
</div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
flask_process = None


def find_free_port():
    for port in PORT_RANGE:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((HOST, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port in {PORT_RANGE.start}-{PORT_RANGE.stop - 1}")


def wait_for_server(port, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((HOST, port), timeout=2):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


def kill_flask():
    global flask_process
    if flask_process and flask_process.poll() is None:
        log.info("Shutting down Flask server...")
        try:
            os.killpg(os.getpgid(flask_process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            flask_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.warning("Flask didn't stop gracefully, force killing...")
            try:
                os.killpg(os.getpgid(flask_process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        flask_process = None
        log.info("Flask server stopped.")


atexit.register(kill_flask)


def start_flask(port):
    global flask_process

    env = os.environ.copy()
    env["PORT"] = str(port)
    env["FLASK_ENV"] = "production"

    venv_python = os.path.join(PROJECT_DIR, "venv", "bin", "python3")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable
    log.info(f"Using Python: {python_exe}")

    flask_process = subprocess.Popen(
        [python_exe, "app.py"],
        cwd=PROJECT_DIR,
        env=env,
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def _stream_output():
        for line in iter(flask_process.stdout.readline, b""):
            try:
                log.info("[flask] " + line.decode("utf-8", errors="replace").rstrip())
            except Exception:
                pass

    threading.Thread(target=_stream_output, daemon=True).start()
    return flask_process


# ---------------------------------------------------------------------------
# Main — window opens FIRST, then Flask starts in background
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info(f"OCR Butterfly launcher starting")
    log.info(f"Project dir: {PROJECT_DIR}")
    log.info(f"Python: {sys.executable}")
    log.info("=" * 60)

    # Import pywebview early so we fail fast
    try:
        import webview

        webview_version = getattr(webview, "__version__", None)
        if not webview_version:
            try:
                webview_version = version("pywebview")
            except PackageNotFoundError:
                webview_version = "unknown"
        log.info(f"pywebview {webview_version} loaded")
    except ImportError as e:
        log.error(f"pywebview not available: {e}")
        # Show native macOS error dialog
        os.system(
            'osascript -e \'display dialog "pywebview is not installed.\\n\\n'
            'Run this in Terminal:\\n'
            'cd \\\"' + PROJECT_DIR.replace('"', '\\\\"') + '\\\"\\n'
            'source venv/bin/activate\\n'
            'pip install pywebview\\n\\n'
            'Then try again." buttons {"OK"} default button "OK" '
            'with icon stop with title "OCR Butterfly"\''
        )
        sys.exit(1)

    # Find a free port
    try:
        port = find_free_port()
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)

    log.info(f"Using port {port}")

    # Create the window IMMEDIATELY with a loading screen
    # This stops macOS from bouncing the icon endlessly
    window = webview.create_window(
        title=APP_TITLE,
        html=LOADING_HTML,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=(MIN_WIDTH, MIN_HEIGHT),
        resizable=True,
        text_select=True,
        zoomable=True,
    )

    def _boot_server_then_navigate():
        """Runs in a thread: start Flask, wait, then redirect the window."""
        try:
            log.info("Starting Flask server...")

            # Update loading screen: server starting
            try:
                window.evaluate_js("""
                    document.getElementById('step-server').className = 'active';
                    document.getElementById('statusText').textContent = 'Starting server...';
                """)
            except Exception:
                pass

            start_flask(port)

            # Update loading screen: model loading
            try:
                window.evaluate_js("""
                    document.getElementById('step-server').className = 'done';
                    document.getElementById('step-server').textContent = '✓ Server started';
                    document.getElementById('step-model').className = 'active';
                    document.getElementById('step-model').textContent = '● Loading OCR model...';
                    document.getElementById('statusText').textContent = 'Loading OCR model (this may take a minute)...';
                """)
            except Exception:
                pass

            log.info(f"Waiting for server on {HOST}:{port}...")
            if not wait_for_server(port, timeout=180):
                log.error("Server failed to start within timeout!")
                try:
                    window.evaluate_js("""
                        document.getElementById('statusText').textContent = 'Server failed to start. Check launcher.log for details.';
                        document.getElementById('spinner').style.display = 'none';
                        document.getElementById('step-model').className = '';
                        document.getElementById('step-model').style.color = '#ef4444';
                        document.getElementById('step-model').textContent = '✗ Server startup failed';
                    """)
                except Exception:
                    pass
                return

            log.info("Server ready! Navigating to app...")

            # Update loading screen: ready
            try:
                window.evaluate_js("""
                    document.getElementById('step-model').className = 'done';
                    document.getElementById('step-model').textContent = '✓ Model loaded';
                    document.getElementById('step-ready').className = 'done';
                    document.getElementById('step-ready').textContent = '✓ Ready!';
                    document.getElementById('statusText').textContent = 'Opening app...';
                """)
            except Exception:
                pass

            time.sleep(0.5)  # Brief pause so user sees the "Ready" state
            window.load_url(f"http://{HOST}:{port}/?t={int(time.time())}")
            log.info("Window navigated to app")

        except Exception as e:
            log.exception(f"Boot thread error: {e}")

    # Start the boot sequence AFTER the window is shown
    def _on_shown():
        boot_thread = threading.Thread(target=_boot_server_then_navigate, daemon=True)
        boot_thread.start()

    # webview.start() blocks until the window is closed
    webview.start(
        func=_on_shown,
        debug=False,
        private_mode=True,
    )

    # Window closed — clean up
    log.info("Window closed. Shutting down...")
    kill_flask()


if __name__ == "__main__":
    main()
