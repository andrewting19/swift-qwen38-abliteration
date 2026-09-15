import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/summarize_saved_output_judgments.py"
SPEC = importlib.util.spec_from_file_location("summarize_saved_output_judgments", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_judges(root: Path, key: str, successes: list[bool], mode: str) -> None:
    for group in MODULE.GROUPS:
        write_jsonl(
            root / "harmbench" / key / f"{group}.jsonl",
            [
                {
                    "id": index,
                    "judgment": {
                        "behavior_success": success,
                        "parse_error": False,
                    },
                }
                for index, success in enumerate(successes)
            ],
        )
        write_jsonl(
            root / "mode" / key / f"{group}.jsonl",
            [
                {
                    "id": index,
                    "judgment": {
                        "mode": mode,
                        "quality": "CLEAN",
                        "parse_error": False,
                    },
                }
                for index in range(len(successes))
            ],
        )


def test_summary_ranks_candidate_against_sibling_base(tmp_path: Path) -> None:
    manifest = {
        "items": [
            {"key": "base-key", "aliases": ["screen/raw/base"]},
            {"key": "candidate-key", "aliases": ["screen/raw/candidate"]},
        ]
    }
    input_manifest = tmp_path / "input.json"
    input_manifest.write_text(json.dumps(manifest))
    write_judges(tmp_path, "base-key", [False, False], "HARD_REFUSAL")
    write_judges(tmp_path, "candidate-key", [True, False], "PARTIAL")
    (tmp_path / "harmbench/manifest.json").write_text("{}")
    (tmp_path / "mode/manifest.json").write_text("{}")

    report = MODULE.summarize(
        input_manifest, tmp_path / "harmbench", tmp_path / "mode"
    )

    candidate = report["ranked_candidates"][0]
    assert candidate["alias"] == "screen/raw/candidate"
    assert candidate["minimum_harmbench_success_rate"] == 0.5
    assert candidate["minimum_substantive_rate"] == 1.0
    assert candidate["total_broken_count"] == 0


def test_absolute_summary_accepts_legacy_aliases_without_raw_base(
    tmp_path: Path,
) -> None:
    manifest = {
        "items": [
            {"key": "candidate-key", "aliases": ["legacy-pilot/rank6"]},
        ]
    }
    input_manifest = tmp_path / "input.json"
    input_manifest.write_text(json.dumps(manifest))
    write_judges(tmp_path, "candidate-key", [True, False], "PARTIAL")
    (tmp_path / "harmbench/manifest.json").write_text("{}")
    (tmp_path / "mode/manifest.json").write_text("{}")

    report = MODULE.summarize(
        input_manifest,
        tmp_path / "harmbench",
        tmp_path / "mode",
        absolute=True,
    )

    candidate = report["ranked_candidates"][0]
    assert candidate["alias"] == "legacy-pilot/rank6"
    assert candidate["minimum_harmbench_success_rate"] == 0.5
    assert candidate["minimum_substantive_rate"] == 1.0
    assert "minimum_harmbench_rate_change" not in candidate
    assert report["ranking_mode"] == "absolute"
