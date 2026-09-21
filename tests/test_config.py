import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swift_abliteration.architecture import expected_writer_count
from swift_abliteration.config import load_config


class ConfigTests(unittest.TestCase):
    def test_orca_scope_has_131_tensors(self):
        cfg = load_config("configs/orca_style_full.toml")
        self.assertEqual(expected_writer_count(cfg), 131)
        self.assertEqual(cfg.direction.layer, 38)

    def test_band_scope_has_68_tensors(self):
        cfg = load_config("configs/huihui_band.toml")
        self.assertEqual(expected_writer_count(cfg), 68)
        self.assertEqual((cfg.edit.first_layer, cfg.edit.last_layer), (18, 51))

    def test_rank6_release_scope_has_128_tensors(self):
        cfg = load_config("configs/release_rank6.toml")
        self.assertEqual(cfg.direction.rank, 6)
        self.assertEqual(expected_writer_count(cfg), 128)
        self.assertFalse(cfg.edit.include_embedding)
        self.assertFalse(cfg.edit.include_mtp)


if __name__ == "__main__":
    unittest.main()
