#!/usr/bin/env python3
"""Serve only the local dashboard and its safe status snapshot."""

from __future__ import annotations

import argparse
import http.server
import mimetypes
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS = ROOT / "runs/gpu/20260914-a100-51065040/status.json"


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """Serve two explicit files so other repository data is not web-accessible."""

    dashboard_path: Path
    status_path: Path

    def do_HEAD(self) -> None:
        self._serve(send_body=False)

    def do_GET(self) -> None:
        self._serve(send_body=True)

    def _serve(self, *, send_body: bool) -> None:
        request_path = urlsplit(self.path).path
        if request_path in {"/", "/dashboard", "/dashboard/", "/dashboard/index.html"}:
            path = self.dashboard_path
            cache_control = "no-cache"
        elif request_path == "/status.json":
            path = self.status_path
            cache_control = "no-store"
        else:
            self.send_error(http.HTTPStatus.NOT_FOUND)
            return

        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(http.HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(http.HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if send_body:
            self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    args = parser.parse_args()
    handler = type(
        "ConfiguredDashboardHandler",
        (DashboardHandler,),
        {"dashboard_path": ROOT / "dashboard/index.html", "status_path": args.status.resolve()},
    )
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Dashboard: http://{args.host}:{args.port}/")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
