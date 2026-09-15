from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "probe_answer_conditions.py"
SPEC = importlib.util.spec_from_file_location("probe_answer_conditions", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_condition_names_and_prompts_are_fixed() -> None:
    assert tuple(MODULE.CONDITIONS) == ("base", "direct_research", "direct_manual")
    assert all(value.strip() for value in MODULE.CONDITIONS.values())


def test_read_prompt_slice_preserves_ids_without_returning_other_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prompts.jsonl"
    rows = [
        {"id": 10, "text": "one"},
        {"id": 11, "text": "two"},
        {"id": 12, "text": "three"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    selected, digest = MODULE.read_prompt_slice(path, 1, 1)

    assert selected == [{"id": 11, "text": "two"}]
    assert len(digest) == 64


def test_atomic_jsonl_returns_file_digest(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    digest = MODULE.atomic_jsonl(path, [{"id": 1, "value": "ok"}])

    assert path.is_file()
    assert len(digest) == 64
