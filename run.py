#!/usr/bin/env python
"""
Run script for the XAI Chat Interface.

This script starts both the backend API server and serves the frontend static files.
"""

import os
import sys
import logging
import argparse
import subprocess
import webbrowser
import time
from threading import Thread

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the XAI Chat Interface")

    parser.add_argument('--port', type=int, default=5000,
                       help='Port for the API server (default: 5000)')
    parser.add_argument('--no-browser', action='store_true',
                       help='Do not open a browser window automatically')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B",
                       help='Model name or path')
    parser.add_argument('--skip-kg', action='store_true',
                       help='Skip Knowledge Graph processing')

    return parser.parse_args()

def start_backend(port, debug, model, skip_kg):
    """Start the backend API server."""
    logger.info(f"Starting backend API server on port {port}...")

    env = os.environ.copy()
    env["FLASK_APP"] = "backend.api_router"

    cmd = [
        sys.executable, "-m", "flask", "run",
        "--host", "0.0.0.0",
        "--port", str(port)
    ]

    if debug:
        env["FLASK_DEBUG"] = "1"

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start backend: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Backend server stopped")

def open_browser(port, delay=2):
    """Open browser after a delay."""
    def _open_browser():
        time.sleep(delay)  # Wait for backend to start
        url = f"http://localhost:{port}"
        logger.info(f"Opening browser at {url}")
        webbrowser.open(url)

    browser_thread = Thread(target=_open_browser)
    browser_thread.daemon = True
    browser_thread.start()

def run():
    """Run the XAI Chat Interface."""
    args = parse_args()

    # Check if the frontend directory exists
    if not os.path.exists("frontend"):
        logger.error("Frontend directory not found. Make sure you're running from the project root.")
        sys.exit(1)

    # Check if the backend exists
    if not os.path.exists("backend"):
        logger.error("Backend directory not found. Make sure you're running from the project root.")
        sys.exit(1)

    # Open browser if not disabled
    if not args.no_browser:
        open_browser(args.port)

    # Start backend
    start_backend(args.port, args.debug, args.model, args.skip_kg)

if __name__ == "__main__":
    run()
