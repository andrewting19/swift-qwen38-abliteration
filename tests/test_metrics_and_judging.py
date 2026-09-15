import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swift_abliteration.judging import (
    deterministic_empty_response_judgment,
    parse_coherence_judgment,
    parse_refusal_judgment,
)
from swift_abliteration.metrics import (
    coherence_means,
    forward_kl_from_logits,
    refusal_rate,
    summarize,
)


class MetricAndJudgeTests(unittest.TestCase):
    def test_empty_response_uses_fixed_invalid_rule(self):
        refusal = deterministic_empty_response_judgment("refusal", "  ")
        self.assertEqual(refusal["label"], "hard_refusal")
        self.assertTrue(refusal["invalid_output"])
        coherence = deterministic_empty_response_judgment("coherence", "")
        self.assertEqual(coherence["completeness"], 1)
        self.assertIsNone(deterministic_empty_response_judgment("refusal", "answer"))

    def test_refusal_rate_uses_hard_and_soft_labels(self):
        rows = [
            {"judgment": {"label": "hard_refusal"}},
            {"judgment": {"label": "soft_refusal"}},
            {"judgment": {"label": "partial_answer"}},
            {"judgment": {"label": "full_answer"}},
        ]
        self.assertEqual(refusal_rate(rows), 0.5)

    def test_coherence_means(self):
        rows = [
            {"judgment": {"readability": 5}},
            {"judgment": {"readability": 3}},
        ]
        self.assertEqual(coherence_means(rows, ["readability"]), {"readability": 4.0})

    def test_identical_logits_have_zero_kl(self):
        logits = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]])
        values = forward_kl_from_logits(logits, logits)
        np.testing.assert_allclose(values, 0, atol=1e-12)

    def test_kl_is_nonnegative(self):
        base = np.array([[2.0, 0.0]])
        edited = np.array([[0.0, 2.0]])
        self.assertGreater(float(forward_kl_from_logits(base, edited)[0]), 0)
        self.assertEqual(summarize(np.array([1.0, 2.0]))["count"], 2)

    def test_refusal_parser(self):
        result = parse_refusal_judgment(
            '{"label":"soft_refusal","confidence":0.8,"reason":"brief"}'
        )
        self.assertEqual(result["label"], "soft_refusal")

    def test_coherence_parser(self):
        result = parse_coherence_judgment(
            '{"readability":5,"relevance":4,"consistency":5,'
            '"non_repetition":5,"completeness":4,"confidence":0.9,"reason":"brief"}'
        )
        self.assertEqual(result["relevance"], 4)


if __name__ == "__main__":
    unittest.main()
