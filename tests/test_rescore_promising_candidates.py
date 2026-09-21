import unittest

from scripts.rescore_promising_candidates import (
    paired_bootstrap_difference,
    wilson_interval,
)


class RevisedScoreTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_rate(self):
        lower, upper = wilson_interval(38, 64)
        self.assertLess(lower, 38 / 64)
        self.assertGreater(upper, 38 / 64)

    def test_identical_paired_scores_pass_noninferiority(self):
        result = paired_bootstrap_difference(
            [1, 0, 1, 1], [1, 0, 1, 1], margin=0.02
        )
        self.assertEqual(result["difference"], 0.0)
        self.assertEqual(result["paired_bootstrap_95_ci"], [0.0, 0.0])
        self.assertEqual(result["noninferiority_result"], "pass")


if __name__ == "__main__":
    unittest.main()
