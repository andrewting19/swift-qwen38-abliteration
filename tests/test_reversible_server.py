import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from swift_abliteration.reversible_server import (
    GenerationResult,
    apply_text_stops,
    make_app,
)


class FakeEngine:
    def generate(self, messages, max_tokens, temperature, top_p, seed):
        assert messages == [{"role": "user", "content": "hello"}]
        assert (max_tokens, temperature, top_p, seed) == (8, 0.0, 1.0, 7)
        return GenerationResult("answer STOP hidden", 3, 4)


class ReversibleServerTests(unittest.TestCase):
    def test_text_stop(self):
        self.assertEqual(apply_text_stops("abc END later", ["END"]), "abc ")

    def test_chat_completion_and_safe_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.jsonl"
            client = TestClient(make_app(FakeEngine(), "test-model", metrics))
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 8,
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 7,
                    "stop": ["STOP"],
                },
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["choices"][0]["message"]["content"], "answer ")
            self.assertEqual(body["usage"]["total_tokens"], 7)
            text = metrics.read_text(encoding="utf-8")
            self.assertNotIn("hello", text)
            self.assertNotIn("answer", text)


if __name__ == "__main__":
    unittest.main()
