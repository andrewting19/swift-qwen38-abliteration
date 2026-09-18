#!/usr/bin/env python3
"""Shift resid-pre layer directions onto the writer that created that state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from swift_abliteration.gpu_support import sha256_file, write_json


def align_resid_pre_to_writers(values: np.ndarray) -> np.ndarray:
    """Assign resid-pre direction i+1 to writer i; keep final row as padding."""
    tensor = np.asarray(values)
    if tensor.ndim != 3 or tensor.shape[0] < 2:
        raise ValueError("Expected [layers, rank, hidden] with at least two layers.")
    return np.concatenate([tensor[1:], tensor[-1:]], axis=0).copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--candidate-key", required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    import torch
    from safetensors.torch import load_file, save_file

    candidates = load_file(str(args.candidate_file), device="cpu")
    if args.candidate_key not in candidates:
        raise KeyError(f"Missing candidate: {args.candidate_key}")
    source = candidates[args.candidate_key].float().numpy()
    aligned = align_resid_pre_to_writers(source).astype(np.float32)
    args.output_dir.mkdir(parents=True)
    tensor_path = args.output_dir / "writer_aligned_layerwise_rank2.safetensors"
    save_file(
        {"writer_aligned_layerwise_rank2": torch.from_numpy(aligned).contiguous()},
        str(tensor_path),
    )
    report = {
        "schema_version": 1,
        "method": "writer layer i uses the direction captured at resid-pre layer i+1",
        "source_candidate_key": args.candidate_key,
        "shape": list(aligned.shape),
        "active_writer_layers": list(range(len(aligned) - 1)),
        "embedding_included": False,
        "final_writer_excluded": True,
        "source_candidate_sha256": sha256_file(args.candidate_file),
        "source_report_sha256": sha256_file(args.source_report),
        "tensor_sha256": sha256_file(tensor_path),
        "uses_final_test": False,
        "checkpoint_saved": False,
        "openai_models_used": False,
        "remote_api_used": False,
        "raw_text_in_report": False,
    }
    write_json(args.output_dir / "writer_aligned_layerwise_rank2_report.json", report)
    print(json.dumps({"status": "complete", "shape": list(aligned.shape)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
