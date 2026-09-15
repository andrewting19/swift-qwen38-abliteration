#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from swift_abliteration.gpu_support import sha256_file, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the fixed quick capability gate."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-accuracy-loss", type=float, default=0.02)
    parser.add_argument("--maximum-valid-rate-loss", type=float, default=0.01)
    args = parser.parse_args()
    source = json.loads(args.summary.read_text(encoding="utf-8"))
    datasets = {}
    for name, values in source["datasets"].items():
        accuracy_delta = values["candidate_accuracy"] - values["base_accuracy"]
        valid_delta = values["candidate_valid_rate"] - values["base_valid_rate"]
        gates = {
            "accuracy": accuracy_delta >= -args.maximum_accuracy_loss - 1e-12,
            "valid_rate": valid_delta >= -args.maximum_valid_rate_loss - 1e-12,
        }
        datasets[name] = {
            **values,
            "accuracy_delta": accuracy_delta,
            "valid_rate_delta": valid_delta,
            "gates": gates,
            "pass": all(gates.values()),
        }
    passed = all(value["pass"] for value in datasets.values())
    output = {
        "source": str(args.summary),
        "source_sha256": sha256_file(args.summary),
        "maximum_accuracy_loss": args.maximum_accuracy_loss,
        "maximum_valid_rate_loss": args.maximum_valid_rate_loss,
        "datasets": datasets,
        "pass": passed,
        "uses_final_test": False,
        "checkpoint_saved": False,
    }
    write_json(args.output, output)
    print(json.dumps({"pass": passed, "datasets": datasets}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
