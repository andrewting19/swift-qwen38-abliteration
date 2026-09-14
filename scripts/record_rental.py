#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from swift_abliteration.gpu_support import system_record, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["start", "end"], required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--hourly-price", type=float)
    parser.add_argument("--instance-id")
    parser.add_argument("--offer-id")
    args = parser.parse_args()
    path = Path(args.record)
    now = datetime.now(UTC)
    if args.phase == "start":
        if path.exists():
            raise FileExistsError(f"Rental record already exists: {path}")
        if args.hourly_price is None or args.hourly_price <= 0:
            raise ValueError("A positive --hourly-price is required at start.")
        record = {
            "started_at_utc": now.isoformat(),
            "hourly_price_usd": args.hourly_price,
            "instance_id": args.instance_id,
            "offer_id": args.offer_id,
            "system_at_start": system_record(),
        }
    else:
        if not path.exists():
            raise FileNotFoundError(f"Rental record does not exist: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        start = datetime.fromisoformat(record["started_at_utc"])
        elapsed_hours = (now - start).total_seconds() / 3600
        record.update(
            {
                "ended_at_utc": now.isoformat(),
                "elapsed_hours": elapsed_hours,
                "estimated_cost_usd": elapsed_hours * record["hourly_price_usd"],
            }
        )
    write_json(path, record)
    safe = {
        key: record[key]
        for key in (
            "started_at_utc",
            "ended_at_utc",
            "elapsed_hours",
            "hourly_price_usd",
            "estimated_cost_usd",
        )
        if key in record
    }
    print(json.dumps(safe, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
