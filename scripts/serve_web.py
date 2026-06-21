#!/usr/bin/env python3
"""Serve the public read-only dashboard with uvicorn.

Reads the SQLite file the bot records to (set STOCKBOT_WEB_DB, or pass --db).
This process never holds Alpaca credentials and exposes no write endpoints.

Usage:
    python scripts/serve_web.py --db ./data/dashboard.db --host 0.0.0.0 --port 8000
"""

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the random-bot dashboard")
    parser.add_argument("--db", type=str, default=None, help="Path to the dashboard SQLite file")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (use 0.0.0.0 behind a proxy)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.db:
        os.environ["STOCKBOT_WEB_DB"] = args.db

    import uvicorn

    # Import path string so uvicorn can manage the app lifecycle.
    uvicorn.run("stockbot.web.app:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
