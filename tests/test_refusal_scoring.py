import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None

from swift_abliteration.refusal_scoring import outcome_masks, resolve_single_token_ids


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"I": [4], "As": [9], "many": [1, 2]}[text]


class RefusalScoringTests(unittest.TestCase):
    def test_resolve_single_token_ids(self):
        self.assertEqual(resolve_single_token_ids(FakeTokenizer()), [4, 9])
        with self.assertRaises(ValueError):
            resolve_single_token_ids(FakeTokenizer(), ["many"])

    def test_outcome_masks_match_positive_negative_rule(self):
        harmful, harmless = outcome_masks(
            np.array([-1.0, 0.0, 2.0]), np.array([-2.0, 0.0, 3.0])
        )
        np.testing.assert_array_equal(harmful, [False, False, True])
        np.testing.assert_array_equal(harmless, [True, False, False])

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_refusal_score_is_log_odds_of_selected_tokens(self):
        from swift_abliteration.refusal_scoring import refusal_scores_from_logits

        probabilities = torch.tensor([[0.6, 0.1, 0.3]], dtype=torch.float64)
        logits = probabilities.log()
        score = refusal_scores_from_logits(logits, [0, 1])
        expected = np.log(0.7) - np.log(0.3)
        self.assertAlmostEqual(float(score[0]), expected, places=6)


if __name__ == "__main__":
    unittest.main()
