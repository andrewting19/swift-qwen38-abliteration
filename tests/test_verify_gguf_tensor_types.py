from pathlib import Path

import pytest

from scripts.verify_gguf_tensor_types import compare_types, read_exact_type_map


def test_read_exact_type_map(tmp_path: Path) -> None:
    path = tmp_path / "types.txt"
    path.write_text(
        "^output\\.weight$=Q5_K\n^blk\\.0\\.ffn_down\\.weight$=IQ3_XXS\n",
        encoding="utf-8",
    )
    assert read_exact_type_map(path) == {
        "output.weight": "Q5_K",
        "blk.0.ffn_down.weight": "IQ3_XXS",
    }


def test_read_exact_type_map_rejects_non_exact_pattern(tmp_path: Path) -> None:
    path = tmp_path / "types.txt"
    path.write_text("blk.*=Q4_K\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact anchored"):
        read_exact_type_map(path)


def test_compare_types_reports_missing_and_wrong() -> None:
    assert compare_types(
        {"a": "Q4_K", "b": "Q5_K", "c": "Q6_K"},
        {"a": "Q4_K", "b": "Q3_K"},
    ) == [("b", "Q5_K", "Q3_K"), ("c", "Q6_K", None)]
