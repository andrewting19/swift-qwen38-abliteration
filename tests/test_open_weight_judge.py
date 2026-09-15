from __future__ import annotations

import unittest

from swift_abliteration.open_weight_judge import (
    parse_harmbench_output,
    parse_wildguard_output,
    render_harmbench_llama_prompt,
    render_wildguard_qwen_prompt,
)


class OpenWeightJudgeTests(unittest.TestCase):
    def test_parse_wildguard_output(self):
        result = parse_wildguard_output(
            "Harmful request: yes\nResponse refusal: no\nHarmful response: yes"
        )
        self.assertEqual(
            result,
            {
                "prompt_harmful": True,
                "response_refusal": False,
                "response_harmful": True,
            },
        )

    def test_parse_is_case_insensitive(self):
        result = parse_wildguard_output(
            "harmful request: No\nresponse refusal: YES\nharmful response: no"
        )
        self.assertTrue(result["response_refusal"])

    def test_parse_requires_all_fields(self):
        with self.assertRaisesRegex(ValueError, "response_harmful"):
            parse_wildguard_output(
                "Harmful request: yes\nResponse refusal: no"
            )

    def test_render_contains_both_inputs(self):
        prompt = render_wildguard_qwen_prompt("safe request", "safe answer")
        self.assertIn("safe request", prompt)
        self.assertIn("safe answer", prompt)

    def test_parse_harmbench_output(self):
        self.assertTrue(parse_harmbench_output(" Yes"))
        self.assertFalse(parse_harmbench_output("no\n"))
        with self.assertRaisesRegex(ValueError, "does not start"):
            parse_harmbench_output("Maybe")

    def test_render_harmbench_contains_both_inputs(self):
        prompt = render_harmbench_llama_prompt("behavior", "generation")
        self.assertIn("behavior", prompt)
        self.assertIn("generation", prompt)


if __name__ == "__main__":
    unittest.main()
