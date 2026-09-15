from scripts.assess_capability_gate import main as capability_main
from scripts.summarize_generation_screen import harmbench_summary


def test_harmbench_summary(tmp_path) -> None:
    path = tmp_path / "judgments.jsonl"
    path.write_text(
        '{"judgment":{"behavior_success":true,"parse_error":false,"empty_output":false}}\n'
        '{"judgment":{"behavior_success":false,"parse_error":false,"empty_output":false}}\n'
    )
    result = harmbench_summary(path)
    assert result["count"] == 2
    assert result["behavior_success_rate"] == 0.5


def test_capability_gate_accepts_two_point_loss(tmp_path, monkeypatch) -> None:
    source = tmp_path / "summary.json"
    output = tmp_path / "gate.json"
    source.write_text(
        '{"datasets":{"cyber":{"base_accuracy":0.5,"candidate_accuracy":0.48,'
        '"base_valid_rate":1.0,"candidate_valid_rate":1.0}}}'
    )
    monkeypatch.setattr(
        "sys.argv", ["assess", "--summary", str(source), "--output", str(output)]
    )
    assert capability_main() == 0
