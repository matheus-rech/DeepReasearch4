"""
Command‑line interface for running the Deep Research application.

This script allows you to start the MCP server, the Streamlit UI, or
both concurrently.  It mirrors the behaviour of the original
``sr_screener/main.py`` but references the modules in the
``full_stack_app`` package.

Usage:

```
python -m full_stack_app.frontend.main [server|ui|both]
```

If no argument is provided, the default is ``both`` which will start
the MCP server on port 8001 in a background thread and then launch
the Streamlit UI on port 8000.  The server is started without
auto‑reloading.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path


def run_server() -> None:
    """Start the MCP server on port 8001."""
    from full_stack_app.backend import mcp_server
    mcp_server.run(host="0.0.0.0", port=8001)


def run_ui() -> None:
    """Start the Streamlit UI on port 8000."""
    app_path = Path(__file__).parent / "app.py"
    # Use subprocess to run Streamlit.  We avoid importing streamlit here
    # to prevent interfering with threading or the event loop used by
    # FastAPI in the MCP server.
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        "8000",
        "--server.address",
        "0.0.0.0",
    ])


def run_both() -> None:
    """Start the MCP server in a background thread and then launch the UI."""
    # Start server thread as daemon so it terminates when the main program exits
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    # Give the server a moment to start up
    time.sleep(2)
    # Launch the Streamlit UI in the main thread
    run_ui()


def main() -> None:
    """Entry point for the CLI.

    Parses the first command‑line argument to determine which component
    to run.  Accepts ``server``, ``ui`` or ``both``.  Defaults to
    ``both`` if no argument is supplied.
    """
    args = sys.argv[1:]
    command = args[0].lower() if args else "both"
    if command == "server":
        run_server()
    elif command == "ui":
        run_ui()
    elif command == "both":
        run_both()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python -m full_stack_app.frontend.main [server|ui|both]")
        sys.exit(1)


if __name__ == "__main__":
    main()