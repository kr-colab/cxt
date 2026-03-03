#!/usr/bin/env python3
"""
Serve the Sphinx-built docs locally (Read the Docs style).

Usage:
  python scripts/serve_docs.py              # serve existing build
  python scripts/serve_docs.py --build     # build then serve
  python scripts/serve_docs.py --port 8080 # custom port

Open http://localhost:8000 (or the chosen port) in your browser.
"""

import argparse
import os
import subprocess
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Run 'make html' in docs/ before serving")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--bind", default="127.0.0.1", help="Address to bind (default: 127.0.0.1)")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    docs_dir = os.path.join(repo_root, "docs")
    html_dir = os.path.join(docs_dir, "build", "html")

    if args.build:
        print("Building docs (make -C docs html)...")
        r = subprocess.run(["make", "html"], cwd=docs_dir)
        if r.returncode != 0:
            sys.exit(r.returncode)
        print("Build done.\n")

    if not os.path.isdir(html_dir):
        print(f"Docs not found at {html_dir}. Run with --build first.")
        sys.exit(1)

    os.chdir(html_dir)
    server = HTTPServer((args.bind, args.port), SimpleHTTPRequestHandler)
    url = f"http://{args.bind}:{args.port}"
    print(f"Serving docs at {url}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
