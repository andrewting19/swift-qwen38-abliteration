from scripts.summarize_alpha_ladder import build_summary


def test_build_summary_marks_only_joint_passes() -> None:
    screen = {
        "arms": {
            "alpha_low": {
                "target_alpha": 0.5,
                "kl": {"mean": 0.05},
            },
            "alpha_high": {
                "target_alpha": 1.0,
                "kl": {"mean": 0.5},
            },
        }
    }
    judge = {
        "summaries": {
            f"{arm}/{group}": {
                "count": 10,
                "refusal_count": 10 if arm == "alpha_low" else 0,
                "parse_error_count": 0,
                "empty_output_count": 0,
            }
            for arm in ("alpha_low", "alpha_high")
            for group in ("standard_harmful", "matched_harmful")
        }
    }

    result = build_summary(screen, judge)

    assert result["joint_pass_count"] == 0
    assert result["results"][0]["gates"] == {
        "mean_kl": True,
        "refusal_removal": False,
        "joint": False,
    }
    assert result["results"][1]["gates"] == {
        "mean_kl": False,
        "refusal_removal": True,
        "joint": False,
    }
