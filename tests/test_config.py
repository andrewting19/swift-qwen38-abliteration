from pathlib import Path
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
