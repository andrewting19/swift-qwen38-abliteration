from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "judge_harmbench_tree.py"
SPEC = importlib.util.spec_from_file_location("judge_harmbench_tree", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summarize_existing_judgments_accepts_complete_file(tmp_path: Path) -> None:
    source_rows = [{"id": "a"}, {"id": "b"}]
    rows = [
        {
            "id": "a",
            "judgment": {
                "behavior_success": True,
                "parse_error": False,
                "empty_output": False,
            },
        },
        {
            "id": "b",
            "judgment": {
                "behavior_success": False,
                "parse_error": False,
                "empty_output": True,
            },
        },
    ]
    path = tmp_path / "judgments.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    summary = MODULE.summarize_existing_judgments(path, source_rows)

    assert summary is not None
    assert summary["count"] == 2
    assert summary["behavior_success_count"] == 1
    assert summary["empty_output_count"] == 1
    assert summary["resumed"] is True


def test_summarize_existing_judgments_rejects_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "judgments.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "a",
                "judgment": {
                    "behavior_success": True,
                    "parse_error": False,
                    "empty_output": False,
                },
            }
        )
        + "\n"
    )

    assert MODULE.summarize_existing_judgments(
        path, [{"id": "a"}, {"id": "b"}]
    ) is None
