#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from swift_abliteration.architecture import hf_json, validate_public_metadata
from swift_abliteration.config import load_config
from swift_abliteration.gpu_support import require_large_gpu

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def check_prompt_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    checked = 0
    records = manifest.get("files", manifest)
    for name, record in records.items():
        candidate = Path(record.get("path", name))
        if not candidate.is_absolute() and not candidate.exists():
            candidate = path.parent / name
        if not candidate.exists() or sha256(candidate) != record["sha256"]:
            raise RuntimeError(f"Prompt artifact check failed: {candidate}")
        checked += 1
    return {"path": str(path), "files_checked": checked}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    checks = {}
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tests.returncode:
        raise RuntimeError("Unit tests failed. Run them directly for details.")
    checks["unit_tests"] = "pass"
    checks["packages"] = {
        name: importlib.metadata.version(name)
        for name in (
            "accelerate",
            "datasets",
            "lm-eval",
            "pillow",
            "safetensors",
            "torch",
            "torchvision",
            "transformers",
        )
    }
    checks["torch_import"] = torch.__version__
    if args.require_gpu:
        require_large_gpu()
        checks["large_bf16_gpu"] = "pass"
    for config_path in ("configs/orca_style_full.toml", "configs/huihui_band.toml"):
        cfg = load_config(ROOT / config_path)
        model_config = hf_json(cfg.model.id, cfg.model.revision, "config.json")
        index = hf_json(
            cfg.model.id, cfg.model.revision, "model.safetensors.index.json"
        )
        report = validate_public_metadata(cfg, model_config, index)
        checks[config_path] = {"planned_tensors": report["planned_tensor_count"]}
    base_cfg = load_config(ROOT / "configs/orca_style_full.toml")
    transformer_cfg = AutoConfig.from_pretrained(
        base_cfg.model.id, revision=base_cfg.model.revision
    )
    model_class = AutoModelForImageTextToText._model_mapping[type(transformer_cfg)]
    processor = AutoProcessor.from_pretrained(
        base_cfg.model.id, revision=base_cfg.model.revision
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "test"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    checks["transformers_runtime"] = {
        "config_class": type(transformer_cfg).__name__,
        "model_class": model_class.__name__,
        "processor_class": type(processor).__name__,
        "tokenizer_length": len(tokenizer),
    }
    checks["standard_prompts"] = check_prompt_manifest(
        ROOT / "data/prepared/manifest.json"
    )
    checks["matched_prompts"] = check_prompt_manifest(
        ROOT / "data/prepared/matched/manifest.json"
    )
    checks["benchmarks"] = check_prompt_manifest(ROOT / "benchmarks/data/manifest.json")
    checks["wmdp"] = check_prompt_manifest(ROOT / "benchmarks/data/wmdp_manifest.json")
    required = [
        "scripts/capture_activations.py",
        "scripts/analyze_directions.py",
        "scripts/screen_directions.py",
        "src/swift_abliteration/intervention.py",
        "src/swift_abliteration/checkpoint_edit.py",
        "scripts/apply_weight_edit.py",
        "scripts/generate_eval.py",
        "scripts/judge_outputs.py",
        "scripts/prepare_wmdp.py",
        "scripts/evaluate_wmdp.py",
        "scripts/summarize_judgments.py",
        "scripts/record_rental.py",
        "scripts/run_lm_eval.sh",
        "infra/serve_transformers.sh",
        "infra/vast/RUNBOOK.md",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    if missing:
        raise RuntimeError("Missing required files: " + ", ".join(missing))
    checks["required_files"] = len(required)
    report = {"ready_for_gpu_rental": True, "checks": checks}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
