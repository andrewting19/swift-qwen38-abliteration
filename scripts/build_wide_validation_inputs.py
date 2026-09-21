#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file, save_file


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build non-final XSTest data and a two-candidate direction bundle."
    )
    parser.add_argument("--rank6", type=Path, required=True)
    parser.add_argument("--rank2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch

    partitions = [
        Path("data/prepared/xstest-safe/selection.jsonl"),
        Path("data/prepared/xstest-safe/validation.jsonl"),
        Path("data/prepared/xstest-safe/audit_remainder.jsonl"),
    ]
    rows = []
    seen_ids = set()
    for partition in partitions:
        for row in read_rows(partition):
            identifier = row["id"]
            if identifier in seen_ids:
                raise ValueError(f"Duplicate XSTest id: {identifier}")
            seen_ids.add(identifier)
            rows.append(row)
    if len(rows) != 186:
        raise ValueError(f"Expected 186 non-final XSTest rows, found {len(rows)}")

    rank6_source = load_file(str(args.rank6), device="cpu")
    rank6_rows = torch.stack(
        [rank6_source[f"iterative_direction_{index}"].float() for index in range(1, 7)]
    )
    q, r = torch.linalg.qr(rank6_rows.transpose(0, 1), mode="reduced")
    if int(torch.linalg.matrix_rank(r).item()) != 6:
        raise ValueError("Rank-6 source directions are linearly dependent.")
    rank6_basis = q.transpose(0, 1).contiguous()

    rank2_source = load_file(str(args.rank2), device="cpu")
    rank2_basis = rank2_source["harmbench_complementary_rank2"].float().contiguous()
    if rank2_basis.shape != (2, 5120):
        raise ValueError(f"Unexpected rank-2 shape: {tuple(rank2_basis.shape)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_path = args.output_dir / "xstest-safe-development-186.jsonl"
    safe_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    safe_path.write_text(safe_payload, encoding="utf-8")
    directions_path = args.output_dir / "wide-validation-directions.safetensors"
    save_file(
        {
            "iterative_rank6": rank6_basis,
            "complementary_rank2": rank2_basis,
        },
        str(directions_path),
    )
    report = {
        "safe_path": str(safe_path),
        "safe_count": len(rows),
        "safe_sha256": hashlib.sha256(safe_payload.encode()).hexdigest(),
        "final_test_excluded": True,
        "direction_path": str(directions_path),
        "direction_sha256": hashlib.sha256(directions_path.read_bytes()).hexdigest(),
        "keys": {"iterative_rank6": 6, "complementary_rank2": 2},
    }
    report_path = args.output_dir / "manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
