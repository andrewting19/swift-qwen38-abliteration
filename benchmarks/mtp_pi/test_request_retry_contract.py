from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from abliteration_station.config import validate_config
from abliteration_station.controller import Controller
from abliteration_station.errors import LifecycleError


class _Response:
    def __init__(self, value: object):
        self._payload = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return io.BytesIO(self._payload)

    def __exit__(self, *_args):
        return False


class RequestRetryContractTest(unittest.TestCase):
    def _controller(self, root: Path, **overrides) -> Controller:
        key = root / "key"
        key.write_text("benchmark-key\n", encoding="utf-8")
        config = {
            "provider_order": ["vast"],
            "providers": {"vast": {}},
            "model": {"id": "test-model", "context_size": 4096},
            "inference_key_file": str(key),
            **overrides,
        }
        return Controller(config)

    @staticmethod
    def _http_error(code: int) -> urllib.error.HTTPError:
        return urllib.error.HTTPError(
            "https://model.test/v1/models", code, "failure", {}, None
        )

    def test_transient_http_error_retries_and_preserves_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            controller = self._controller(
                Path(temp), model_gate_attempts=3, model_gate_backoff_seconds=0.25
            )
            effects = [self._http_error(503), self._http_error(429), _Response({"ok": True})]
            with patch("urllib.request.urlopen", side_effect=effects) as urlopen, patch(
                "abliteration_station.controller.time.sleep"
            ) as sleep:
                result = controller._request_json(
                    "https://model.test", "/v1/chat/completions", body={"x": 1}
                )

            self.assertEqual(result, {"ok": True})
            self.assertEqual(urlopen.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(sleep.call_args_list[0].args, (0.25,))
            requests = [call.args[0] for call in urlopen.call_args_list]
            self.assertEqual({request.full_url for request in requests}, {"https://model.test/v1/chat/completions"})
            self.assertTrue(all(request.data == b'{"x": 1}' for request in requests))
            self.assertTrue(all(request.headers["Authorization"] == "Bearer benchmark-key" for request in requests))

    def test_permanent_http_error_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            controller = self._controller(Path(temp), model_gate_attempts=5)
            with patch("urllib.request.urlopen", side_effect=self._http_error(401)) as urlopen, patch(
                "abliteration_station.controller.time.sleep"
            ) as sleep:
                with self.assertRaisesRegex(LifecycleError, "1 attempt"):
                    controller._request_json("https://model.test", "/v1/models")
            self.assertEqual(urlopen.call_count, 1)
            sleep.assert_not_called()

    def test_invalid_json_is_not_retried(self) -> None:
        class InvalidResponse:
            def __enter__(self):
                return io.BytesIO(b"not-json")

            def __exit__(self, *_args):
                return False

        with tempfile.TemporaryDirectory() as temp:
            controller = self._controller(Path(temp), model_gate_attempts=4)
            with patch("urllib.request.urlopen", return_value=InvalidResponse()) as urlopen, patch(
                "abliteration_station.controller.time.sleep"
            ) as sleep:
                with self.assertRaises(LifecycleError):
                    controller._request_json("https://model.test", "/v1/models")
            self.assertEqual(urlopen.call_count, 1)
            sleep.assert_not_called()

    def test_exhausted_transient_error_reports_attempt_count_and_chains(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            controller = self._controller(
                Path(temp), model_gate_attempts=2, model_gate_backoff_seconds=0
            )
            failure = urllib.error.URLError("not ready")
            with patch("urllib.request.urlopen", side_effect=failure), patch(
                "abliteration_station.controller.time.sleep"
            ):
                with self.assertRaisesRegex(LifecycleError, r"https://model\.test.*2 attempts") as caught:
                    controller._request_json("https://model.test", "/v1/models")
            self.assertIs(caught.exception.__cause__, failure)

    def test_retry_configuration_contract(self) -> None:
        valid = {
            "provider_order": ["vast"],
            "providers": {"vast": {}},
            "model": {"id": "test-model", "context_size": 4096},
            "model_gate_attempts": 1,
            "model_gate_backoff_seconds": 0.0,
        }
        validate_config(valid)
        for bad_attempts in (True, 0, -1, 1.5, "3"):
            with self.subTest(model_gate_attempts=bad_attempts), self.assertRaises(ValueError):
                validate_config({**valid, "model_gate_attempts": bad_attempts})
        for bad_backoff in (True, -0.1, "1"):
            with self.subTest(model_gate_backoff_seconds=bad_backoff), self.assertRaises(ValueError):
                validate_config({**valid, "model_gate_backoff_seconds": bad_backoff})


if __name__ == "__main__":
    unittest.main()
