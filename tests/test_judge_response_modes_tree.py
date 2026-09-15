from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "judge_response_modes_tree.py"
SPEC = importlib.util.spec_from_file_location("judge_response_modes_tree", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summarize_existing_judgments_accepts_complete_file(tmp_path: Path) -> None:
    source_rows = [{"id": "a"}, {"id": "b"}]
    rows = [
        {
            "id": "a",
            "judgment": {
                "mode": "DIRECT",
                "quality": "CLEAN",
                "parse_error": False,
            },
        },
        {
            "id": "b",
            "judgment": {
                "mode": "SOFT_REFUSAL",
                "quality": "BROKEN",
                "parse_error": False,
            },
        },
    ]
    path = tmp_path / "judgments.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    summary = MODULE.summarize_existing_judgments(path, source_rows)

    assert summary is not None
    assert summary["count"] == 2
    assert summary["mode_counts"]["DIRECT"] == 1
    assert summary["broken_count"] == 1
    assert summary["resumed"] is True


def test_summarize_existing_judgments_rejects_wrong_ids(tmp_path: Path) -> None:
    path = tmp_path / "judgments.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "a",
                "judgment": {
                    "mode": "DIRECT",
                    "quality": "CLEAN",
                    "parse_error": False,
                },
            }
        )
        + "\n"
    )

    assert MODULE.summarize_existing_judgments(path, [{"id": "b"}]) is None
