#!/usr/bin/env python3
"""Serve the local dashboard and explicit experiment review files."""

from __future__ import annotations

import argparse
import http.server
import mimetypes
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS = ROOT / "runs/gpu/20260914-a100-51065040/status.json"

BASE_RUN = ROOT / "runs/gpu/20260914-a100-51065040"
FINAL_RUN = ROOT / "runs/gpu/20260915-a100-51081304"


def comparison_files() -> dict[str, Path]:
    """Return the fixed route allowlist for the before/after review."""
    files: dict[str, Path] = {}
    groups = (
        "standard_harmful",
        "matched_harmful",
        "standard_harmless",
        "matched_harmless",
    )
    for group in groups:
        files[f"/comparison-data/{group}/base.jsonl"] = (
            BASE_RUN / f"screen-layer38/base/{group}.jsonl"
        )
        after_root = (
            FINAL_RUN / "candidate-r123456-harmful256/alpha_1_000"
            if group.endswith("harmful") and not group.endswith("harmless")
            else FINAL_RUN / "candidate-r123456-full64/candidate_r123456_full64"
        )
        files[f"/comparison-data/{group}/after.jsonl"] = after_root / f"{group}.jsonl"
        files[f"/comparison-data/{group}/base-wildguard.jsonl"] = (
            BASE_RUN / f"judgments_open_weight_wildguard/base/{group}.jsonl"
        )
        is_harmful = group.endswith("harmful") and not group.endswith("harmless")
        after_wildguard_root = (
            FINAL_RUN / "candidate-r123456-harmful256-wildguard/alpha_1_000"
            if is_harmful
            else FINAL_RUN
            / "candidate-r123456-full64-wildguard/candidate_r123456_full64"
        )
        files[f"/comparison-data/{group}/after-wildguard.jsonl"] = (
            after_wildguard_root / f"{group}.jsonl"
        )
        if is_harmful:
            files[f"/comparison-data/{group}/base-harmbench.jsonl"] = (
                BASE_RUN / f"judgments_open_weight_harmbench/base/{group}.jsonl"
            )
            files[f"/comparison-data/{group}/after-harmbench.jsonl"] = (
                FINAL_RUN
                / "candidate-r123456-harmful256-harmbench/alpha_1_000"
                / f"{group}.jsonl"
            )
    return files


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """Serve an explicit allowlist so other repository data is not web-accessible."""

    dashboard_path: Path
    status_path: Path
    comparison_path: Path
    comparison_data: dict[str, Path]

    def do_HEAD(self) -> None:
        self._serve(send_body=False)

    def do_GET(self) -> None:
        self._serve(send_body=True)

    def _serve(self, *, send_body: bool) -> None:
        request_path = urlsplit(self.path).path
        if request_path in {"/", "/dashboard", "/dashboard/", "/dashboard/index.html"}:
            path = self.dashboard_path
            cache_control = "no-cache"
        elif request_path in {"/compare", "/compare/", "/comparison.html"}:
            path = self.comparison_path
            cache_control = "no-cache"
        elif request_path == "/status.json":
            path = self.status_path
            cache_control = "no-store"
        elif request_path in self.comparison_data:
            path = self.comparison_data[request_path]
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
        {
            "dashboard_path": ROOT / "dashboard/index.html",
            "status_path": args.status.resolve(),
            "comparison_path": ROOT / "dashboard/comparison.html",
            "comparison_data": comparison_files(),
        },
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
