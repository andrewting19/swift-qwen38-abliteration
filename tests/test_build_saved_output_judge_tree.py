import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/build_saved_output_judge_tree.py"
SPEC = importlib.util.spec_from_file_location("build_saved_output_judge_tree", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_rows(path: Path, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": index, "prompt": f"prompt {index}", "response": f"{prefix} {index}"}
        for index in range(2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_pair(root: Path, screen: str, arm: str, prefix: str) -> None:
    for group in MODULE.GROUPS:
        write_rows(root / screen / "raw" / arm / f"{group}.jsonl", prefix)


def test_discovery_deduplicates_identical_output_pairs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_pair(run_root, "screen-a", "arm-a", "same")
    write_pair(run_root, "screen-b", "arm-b", "same")
    write_pair(run_root, "screen-c", "arm-c", "different")

    manifest = MODULE.discover(run_root)

    assert manifest["alias_count"] == 3
    assert manifest["unique_candidate_count"] == 2
    assert manifest["judgment_count"] == 8
    assert sorted(len(item["aliases"]) for item in manifest["items"]) == [1, 2]
    assert manifest["raw_text_in_manifest"] is False


def test_materialized_tree_reuses_matching_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    output_root = tmp_path / "judge-input"
    write_pair(run_root, "screen", "arm", "answer")
    manifest = MODULE.discover(run_root)

    MODULE.materialize(run_root, output_root, manifest)
    MODULE.materialize(run_root, output_root, manifest)

    saved = json.loads((output_root / "manifest.json").read_text())
    assert saved["source_fingerprint"] == manifest["source_fingerprint"]
    key = manifest["items"][0]["key"]
    assert (output_root / key / "standard_harmful.jsonl").is_file()


def test_all_pairs_finds_non_raw_responses_and_skips_judgments(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    for group in MODULE.GROUPS:
        write_rows(run_root / "pilot" / "arm" / f"{group}.jsonl", "answer")
        judgment_path = run_root / "judge" / "arm" / f"{group}.jsonl"
        judgment_path.parent.mkdir(parents=True, exist_ok=True)
        judgment_path.write_text(json.dumps({"id": 1, "judgment": {}}) + "\n")

    manifest = MODULE.discover(run_root, all_pairs=True)

    assert manifest["alias_count"] == 1
    assert manifest["unique_candidate_count"] == 1
    assert manifest["skipped_non_response_pair_count"] == 1
