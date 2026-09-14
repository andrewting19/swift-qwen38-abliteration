import tomllib
import unittest
from pathlib import Path


class MatchedSplitTests(unittest.TestCase):
    def test_pair_ids_are_valid_and_disjoint(self):
        with Path("data/matched_splits.toml").open("rb") as handle:
            spec = tomllib.load(handle)
        direction = spec["direction_pair_indices"]
        evaluation = spec["evaluation_pair_indices"]
        final_test = spec["final_test_pair_indices"]
        self.assertEqual(len(direction), 32)
        self.assertEqual(len(evaluation), 64)
        self.assertEqual(len(final_test), 64)
        self.assertFalse(set(direction) & set(evaluation))
        self.assertFalse(set(direction) & set(final_test))
        self.assertFalse(set(evaluation) & set(final_test))
        self.assertTrue(
            all(
                0 <= index < spec["pair_count"]
                for index in direction + evaluation + final_test
            )
        )


if __name__ == "__main__":
    unittest.main()
