from swift_abliteration.refusal_heuristics import arditi_refusal, xstest_refusal


def test_xstest_rule_checks_only_the_opening() -> None:
    assert xstest_refusal("I cannot help with that.") is True
    assert xstest_refusal("Here is an answer. I cannot add more detail.") is False


def test_xstest_rule_is_case_and_space_insensitive() -> None:
    assert xstest_refusal("  AS AN AI, I cannot do that.") is True


def test_arditi_rule_checks_the_full_response() -> None:
    assert arditi_refusal("First, a warning: I am unable to help with this.") is True
    assert arditi_refusal("Here is the requested explanation.") is False
