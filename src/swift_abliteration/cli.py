from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .architecture import hf_json, validate_public_metadata
from .config import load_config


def _write_report(report: dict, output: str | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


def cmd_preflight(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    model_config = hf_json(cfg.model.id, cfg.model.revision, "config.json")
    index = hf_json(cfg.model.id, cfg.model.revision, "model.safetensors.index.json")
    report = validate_public_metadata(cfg, model_config, index)
    report["full_weights_downloaded"] = False
    _write_report(report, args.output)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    _write_report(asdict(load_config(args.config)), None)
    return 0


def cmd_gpu(args: argparse.Namespace) -> int:
    if not args.acknowledge_large_model_run:
        print(
            "STOP: this command can download about 55.6 GB and load the full model. "
            "Review docs/BUILD_REVIEW.md, then pass --acknowledge-large-model-run.",
            file=sys.stderr,
        )
        return 2
    print(
        "The acknowledgement was received. The full execution stage is intentionally "
        "not started in this local-preparation milestone."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swift-abliterate")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser(
        "preflight", help="Validate public metadata without model weights."
    )
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--output")
    preflight.set_defaults(func=cmd_preflight)
    show = sub.add_parser("show-config")
    show.add_argument("--config", required=True)
    show.set_defaults(func=cmd_show)
    gpu = sub.add_parser(
        "gpu-run", help="Guarded boundary for the later full-model stage."
    )
    gpu.add_argument("--config", required=True)
    gpu.add_argument("--acknowledge-large-model-run", action="store_true")
    gpu.set_defaults(func=cmd_gpu)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
