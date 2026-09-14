import tomllib
import unittest
from pathlib import Path


class StandardSplitTests(unittest.TestCase):
    def test_ids_are_disjoint(self):
        with Path("data/splits.toml").open("rb") as handle:
            spec = tomllib.load(handle)
        for name in ("harmful", "harmless"):
            direction = spec[name]["direction_indices"]
            evaluation = spec[name]["evaluation_indices"]
            final_test = spec[name]["final_test_indices"]
            self.assertEqual(len(direction), 32)
            self.assertEqual(len(evaluation), 64)
            self.assertEqual(len(final_test), 64)
            groups = [set(direction), set(evaluation), set(final_test)]
            self.assertTrue(
                all(
                    len(group) == expected
                    for group, expected in zip(groups, (32, 64, 64), strict=True)
                )
            )
            self.assertFalse(groups[0] & groups[1])
            self.assertFalse(groups[0] & groups[2])
            self.assertFalse(groups[1] & groups[2])


if __name__ == "__main__":
    unittest.main()
