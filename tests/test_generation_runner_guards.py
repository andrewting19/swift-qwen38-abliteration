from pathlib import Path


def test_final_test_paths_follow_capability_pass_branch() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "infra"
        / "vast"
        / "run_generation_search.sh"
    )
    text = path.read_text()
    capability_gate = text.index("generation_capability_exit")
    final_test_path = text.index("data/prepared/final_test_harmful.jsonl")
    assert capability_gate < final_test_path
    assert "if [[ $generation_capability_exit -eq 2 ]]" in text


def test_runner_uses_only_local_open_weight_judge() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "infra"
        / "vast"
        / "run_generation_search.sh"
    )
    text = path.read_text().casefold()
    assert "judge_harmbench_tree.py" in text
    assert "wildguard" not in text
    assert "openai" not in text
