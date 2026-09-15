from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "judge_wildguard_tree.py"
SPEC = importlib.util.spec_from_file_location("judge_wildguard_tree", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_write_judgments_returns_safe_counts(tmp_path: Path) -> None:
    path = tmp_path / "judgments.jsonl"
    judgments = [
        {
            "id": 0,
            "judgment": {
                "response_refusal": True,
                "response_harmful": False,
                "parse_error": False,
                "empty_output": False,
            },
        },
        {
            "id": 1,
            "judgment": {
                "response_refusal": False,
                "response_harmful": True,
                "parse_error": False,
                "empty_output": False,
            },
        },
    ]
    summary = MODULE.write_judgments(path, judgments)
    assert summary["count"] == 2
    assert summary["refusal_count"] == 1
    assert summary["harmful_response_count"] == 1
    assert len(summary["output_sha256"]) == 64
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
