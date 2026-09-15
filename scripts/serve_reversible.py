#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

from swift_abliteration.config import load_config
from swift_abliteration.gpu_support import (
    require_acknowledgement,
    require_large_gpu,
    sha256_file,
    system_record,
    write_json,
)
from swift_abliteration.intervention import weight_equivalent_ablation_hooks
from swift_abliteration.live_model import validate_live_model
from swift_abliteration.reversible_server import (
    TransformersGenerationEngine,
    make_app,
)


def parse_layers(value: str) -> list[int]:
    layers: list[int] = []
    for part in value.split(","):
        if "-" in part:
            first, last = (int(item) for item in part.split("-", maxsplit=1))
            layers.extend(range(first, last + 1))
        elif part.strip():
            layers.append(int(part))
    return list(dict.fromkeys(layers))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/orca_style_full.toml")
    parser.add_argument("--directions")
    parser.add_argument("--direction-key")
    parser.add_argument("--intervention-layers")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--served-model-name", default="swift-eval")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    if bool(args.directions) != bool(args.direction_key):
        raise RuntimeError("Pass both --directions and --direction-key, or neither.")
    if args.directions and not args.intervention_layers:
        raise RuntimeError("An intervention layer range is required with a direction.")

    import torch
    import uvicorn
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    processor = AutoProcessor.from_pretrained(cfg.model.id, revision=cfg.model.revision)
    tokenizer = getattr(processor, "tokenizer", processor)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model.id,
        revision=cfg.model.revision,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    validate_live_model(model, cfg)

    hook_context = nullcontext()
    direction_record = None
    if args.directions:
        from safetensors.torch import load_file

        directions = load_file(args.directions, device="cpu")
        if args.direction_key not in directions:
            raise RuntimeError(f"Direction key is not present: {args.direction_key}")
        layers = parse_layers(args.intervention_layers)
        expected_layers = list(range(cfg.edit.first_layer, cfg.edit.last_layer + 1))
        if layers != expected_layers:
            raise RuntimeError(
                "Intervention layers must match the configured checkpoint edit range."
            )
        hook_context = weight_equivalent_ablation_hooks(
            model,
            cfg,
            directions[args.direction_key],
            args.alpha,
            generation_uses_mtp=False,
        )
        direction_record = {
            "file": str(Path(args.directions).resolve()),
            "sha256": sha256_file(args.directions),
            "key": args.direction_key,
            "layers": layers,
            "alpha": args.alpha,
            "type": "weight_equivalent_module_output_projection",
            "embedding_included": cfg.edit.include_embedding,
            "mtp_included": False,
        }

    write_json(
        run_dir / "server_manifest.json",
        {
            "model": cfg.model.id,
            "revision": cfg.model.revision,
            "served_model_name": args.served_model_name,
            "direction": direction_record,
            "system": system_record(),
        },
    )
    print(
        json.dumps(
            {
                "status": "starting",
                "model": args.served_model_name,
                "direction": direction_record,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    engine = TransformersGenerationEngine(model, tokenizer)
    app = make_app(engine, args.served_model_name, run_dir / "request_metrics.jsonl")
    with hook_context:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
