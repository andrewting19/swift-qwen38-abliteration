import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import update_dashboard_status as updater


class DashboardStatusTests(unittest.TestCase):
    def test_count_parser_preserves_safe_path_keys(self):
        with patch.object(updater, "remote_command", return_value=(0, "12 /run/base/standard_harmful.jsonl\n7 /run/candidate/standard_harmful.jsonl\n", "")):
            result = updater.count_remote_jsonl("gpu", ["/run/base/standard_harmful.jsonl", "/run/candidate/standard_harmful.jsonl"], None)
        self.assertEqual(result["/run/base/standard_harmful.jsonl"], 12)
        self.assertEqual(result["/run/candidate/standard_harmful.jsonl"], 7)

    def test_count_command_allows_missing_future_files(self):
        with patch.object(updater, "remote_command", return_value=(0, "12 /run/base/standard_harmful.jsonl\n", "")) as remote:
            result = updater.count_remote_jsonl(
                "gpu",
                ["/run/base/standard_harmful.jsonl", "/run/future/output.jsonl"],
                None,
            )
        self.assertEqual(result, {"/run/base/standard_harmful.jsonl": 12})
        self.assertIn('[ -f "$p" ]', remote.call_args.args[1])
        self.assertTrue(remote.call_args.args[1].endswith("; true"))

    def test_stopped_process_is_reported_without_command_output(self):
        def fake_remote(_host, command, _identity, _timeout, _port=None):
            if command.startswith("ps -eo"):
                return 1, "", ""
            return 0, "A100, 53039, 50\n", ""

        with patch.object(updater, "remote_command", side_effect=fake_remote):
            result = updater.remote_runtime("gpu", None, "screen_directions")
        self.assertFalse(result["process_active"])
        self.assertEqual(result["memory_used_mib"], 53039)

    def test_atomic_write_is_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "status.json"
            updater.write_atomic(target, {"safe": {"count": 25}})
            self.assertEqual(json.loads(target.read_text()), {"safe": {"count": 25}})


if __name__ == "__main__":
    unittest.main()
