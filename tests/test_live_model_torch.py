import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None

from swift_abliteration.config import (
    DirectionSpec,
    EditSpec,
    ExperimentConfig,
    ModelSpec,
)


@unittest.skipIf(torch is None, "torch is not installed")
class LiveModelTests(unittest.TestCase):
    def test_capture_last_positions_uses_resid_pre_and_left_padding(self):
        from types import SimpleNamespace

        from swift_abliteration.live_model import (
            capture_last_positions_resid_pre_multi,
        )

        class Tokenizer:
            padding_side = "right"

            def apply_chat_template(self, messages, **_kwargs):
                return messages[-1]["content"]

            def __call__(self, texts, **_kwargs):
                rows = [[int(value) for value in text.split()] for text in texts]
                width = max(len(row) for row in rows)
                padded = [[0] * (width - len(row)) + row for row in rows]
                return {
                    "input_ids": torch.tensor(padded),
                    "attention_mask": torch.tensor(
                        [[value != 0 for value in row] for row in padded]
                    ),
                }

        class Layer(torch.nn.Module):
            def __init__(self, delta):
                super().__init__()
                self.delta = delta

            def forward(self, hidden):
                return hidden + self.delta

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(16, 1)
                self.embed_tokens.weight.data[:, 0] = torch.arange(16)
                self.layers = torch.nn.ModuleList([Layer(10), Layer(100)])

        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SimpleNamespace(language_model=Backbone())

            def forward(self, input_ids, **_kwargs):
                value = self.model.language_model.embed_tokens(input_ids)
                for layer in self.model.language_model.layers:
                    value = layer(value)
                return SimpleNamespace(last_hidden_state=value)

        tokenizer = Tokenizer()
        captured = capture_last_positions_resid_pre_multi(
            Root(),
            tokenizer,
            ["1 2 3 4", "5 6 7"],
            [0, 1],
            position_count=2,
            batch_size=2,
        )
        torch.testing.assert_close(captured[0][0][:, 0], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(captured[0][1][:, 0], torch.tensor([6.0, 7.0]))
        torch.testing.assert_close(captured[1][0][:, 0], torch.tensor([13.0, 14.0]))
        torch.testing.assert_close(captured[1][1][:, 0], torch.tensor([16.0, 17.0]))
        self.assertEqual(tokenizer.padding_side, "right")

    def test_capability_direction_combiner_accepts_saved_rank2_basis(self):
        import importlib.util

        path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "evaluate_reversible_multiple_choice.py"
        )
        spec = importlib.util.spec_from_file_location("reversible_mc", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        basis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        combined = module.combine_direction_rows([basis])
        torch.testing.assert_close(combined @ combined.T, torch.eye(2))

    def test_batched_generation_returns_responses_and_first_logits(self):
        from types import SimpleNamespace

        from swift_abliteration.live_model import generate_responses_with_first_logits

        class Tokenizer:
            padding_side = "right"
            pad_token_id = 0

            def apply_chat_template(self, messages, **_kwargs):
                return messages[-1]["content"]

            def __call__(self, texts, **_kwargs):
                width = max(len(text) for text in texts)
                rows = [[0] * (width - len(text)) + [1] * len(text) for text in texts]
                return {
                    "input_ids": torch.tensor(rows),
                    "attention_mask": torch.tensor(
                        [[value != 0 for value in row] for row in rows]
                    ),
                }

            def decode(self, token_ids, **_kwargs):
                return " ".join(str(int(value)) for value in token_ids)

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(16, 2)
                self.layers = torch.nn.ModuleList([])

        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = Backbone()

            def generate(self, input_ids, **_kwargs):
                extra = torch.tensor([[7, 8]]).repeat(input_ids.shape[0], 1)
                scores = (torch.arange(16).float().repeat(input_ids.shape[0], 1),)
                return SimpleNamespace(
                    sequences=torch.cat((input_ids, extra), dim=1), scores=scores
                )

        tokenizer = Tokenizer()
        result = generate_responses_with_first_logits(
            Root(), tokenizer, ["a", "bbb", "cc"], 2, batch_size=2
        )
        self.assertEqual(result["responses"], ["7 8", "7 8", "7 8"])
        self.assertEqual(result["first_token_ids"], [7, 7, 7])
        self.assertEqual(len(result["first_step_logits"]), 3)
        self.assertEqual(tokenizer.padding_side, "right")

    def test_activation_addition_hook_is_temporary(self):
        from swift_abliteration.intervention import activation_addition_input_hook

        class Layer(torch.nn.Module):
            def forward(self, hidden_states):
                return hidden_states * 2

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(4, 2)
                self.layers = torch.nn.ModuleList([Layer()])

        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = Backbone()

        model = Root()
        hidden = torch.tensor([[[2.0, 3.0]]])
        direction = torch.tensor([1.0, -1.0])
        before = model.model.language_model.layers[0](hidden)
        with activation_addition_input_hook(model, direction, 0, 0.5) as record:
            during = model.model.language_model.layers[0](hidden)
        after = model.model.language_model.layers[0](hidden)
        torch.testing.assert_close(during, torch.tensor([[[5.0, 5.0]]]))
        torch.testing.assert_close(before, after)
        self.assertEqual(record["layer"], 0)

    def test_in_memory_weight_projection_supports_row_subspaces(self):
        from swift_abliteration.torch_ops import (
            project_embedding_rows_,
            project_output_weight_,
        )

        basis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        output_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        embedding = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        project_output_weight_(output_weight, basis, 1.0, column_chunk=1)
        project_embedding_rows_(embedding, basis, 1.0, row_chunk=1)
        torch.testing.assert_close(output_weight[:2], torch.zeros(2, 2))
        torch.testing.assert_close(embedding[:, :2], torch.zeros(2, 2))
        torch.testing.assert_close(output_weight[2], torch.tensor([5.0, 6.0]))
        torch.testing.assert_close(embedding[:, 2], torch.tensor([3.0, 6.0]))

    def test_weight_equivalent_hooks_match_explicit_projected_weights(self):
        from swift_abliteration.intervention import weight_equivalent_ablation_hooks
        from swift_abliteration.live_model import apply_edit

        class LinearAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.out_proj = torch.nn.Linear(6, 4, bias=True)

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.down_proj = torch.nn.Linear(8, 4, bias=True)

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_attn = LinearAttention()
                self.mlp = MLP()

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(16, 4)
                self.layers = torch.nn.ModuleList([Layer()])

        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = Backbone()

            def get_output_embeddings(self):
                return None

        cfg = ExperimentConfig(
            name="equivalence",
            seed=1,
            model=ModelSpec("fake", "fake", "fake", "fake", 4, 8, 16, 6, 1, 1, 0),
            direction=DirectionSpec(0, 1, "plain", 2, 2, True, "disabled"),
            edit=EditSpec(
                1.0, 0, 0, True, True, True, False, "float32", "bfloat16", True
            ),
        )
        hooked = Root()
        edited = copy.deepcopy(hooked)
        direction = torch.tensor([1.0, 2.0, -1.0, 0.5])
        token_ids = torch.tensor([[1, 4, 7]])
        attention_input = torch.randn(1, 3, 6)
        mlp_input = torch.randn(1, 3, 8)
        with weight_equivalent_ablation_hooks(hooked, cfg, direction) as record:
            hooked_outputs = (
                hooked.model.language_model.embed_tokens(token_ids),
                hooked.model.language_model.layers[0].linear_attn.out_proj(
                    attention_input
                ),
                hooked.model.language_model.layers[0].mlp.down_proj(mlp_input),
            )
        apply_edit(edited, cfg, direction)
        edited_outputs = (
            edited.model.language_model.embed_tokens(token_ids),
            edited.model.language_model.layers[0].linear_attn.out_proj(attention_input),
            edited.model.language_model.layers[0].mlp.down_proj(mlp_input),
        )
        for observed, expected in zip(hooked_outputs, edited_outputs, strict=True):
            torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(record["module_count"], 3)
        self.assertEqual(len(record["bias_modules"]), 2)

    def test_checkpoint_shard_edit_includes_checkpoint_only_mtp(self):
        from swift_abliteration.checkpoint_edit import edit_shard_tensors

        name = "mtp.layers.0.mlp.down_proj.weight"
        untouched_name = "mtp.layers.0.mlp.gate_proj.weight"
        tensors = {
            name: torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
            untouched_name: torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
        }
        before = tensors[untouched_name].clone()
        edited = edit_shard_tensors(
            tensors, {name}, torch.tensor([1.0, 0.0]), alpha=1.0
        )
        self.assertEqual(edited, [name])
        self.assertTrue(torch.equal(tensors[name][0], torch.zeros(2)))
        self.assertTrue(torch.equal(tensors[untouched_name], before))

    def test_direction_key_layer_parser(self):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "scripts" / "screen_directions.py"
        spec = importlib.util.spec_from_file_location("screen_directions", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertEqual(module.direction_layer("matched_layer_38_plain"), 38)

    def test_activation_intervention_is_temporary_and_removes_component(self):
        from swift_abliteration.intervention import activation_ablation_hooks

        class PassThrough(torch.nn.Module):
            def forward(self, hidden):
                return (hidden + torch.tensor([[[3.0, 2.0, 1.0, 0.0]]]), "cache")

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(2, 4)
                self.layers = torch.nn.ModuleList([PassThrough()])

        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = Backbone()

        model = Root()
        hidden = torch.zeros(1, 1, 4)
        direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
        before = model.model.language_model.layers[0](hidden)[0]
        with activation_ablation_hooks(model, direction, [0]):
            during = model.model.language_model.layers[0](hidden)[0]
        after = model.model.language_model.layers[0](hidden)[0]
        self.assertEqual(float(before[0, 0, 0]), 3.0)
        self.assertEqual(float(during[0, 0, 0]), 0.0)
        self.assertTrue(torch.equal(before, after))

    def test_layerwise_weight_hooks_use_distinct_directions(self):
        from swift_abliteration.intervention import (
            layerwise_weight_equivalent_ablation_hooks,
        )

        class LinearAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.out_proj = torch.nn.Linear(2, 2, bias=False)

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.down_proj = torch.nn.Linear(2, 2, bias=False)

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_attn = LinearAttention()
                self.mlp = MLP()

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(4, 2)
                self.layers = torch.nn.ModuleList([Layer(), Layer()])

        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = Backbone()

        cfg = ExperimentConfig(
            name="layerwise",
            seed=1,
            model=ModelSpec("fake", "fake", "fake", "fake", 2, 2, 4, 2, 2, 2, 0),
            direction=DirectionSpec(0, 1, "plain", 2, 2, True, "disabled"),
            edit=EditSpec(
                1.0, 0, 1, True, True, False, False, "float32", "bfloat16", True
            ),
        )
        model = Root()
        for layer in model.model.language_model.layers:
            layer.linear_attn.out_proj.weight.data.copy_(torch.eye(2))
            layer.mlp.down_proj.weight.data.copy_(torch.eye(2))
        value = torch.tensor([[[3.0, 4.0]]])
        directions = {
            0: torch.tensor([1.0, 0.0]),
            1: torch.tensor([0.0, 1.0]),
        }
        with layerwise_weight_equivalent_ablation_hooks(
            model,
            cfg,
            directions,
            attention_alpha=0.5,
            mlp_alpha=0.25,
            attention_layers={0},
            mlp_layers={1},
        ) as record:
            first = model.model.language_model.layers[0].linear_attn.out_proj(value)
            second = model.model.language_model.layers[1].linear_attn.out_proj(value)
            first_mlp = model.model.language_model.layers[0].mlp.down_proj(value)
            second_mlp = model.model.language_model.layers[1].mlp.down_proj(value)
        torch.testing.assert_close(first, torch.tensor([[[1.5, 4.0]]]))
        torch.testing.assert_close(second, torch.tensor([[[3.0, 4.0]]]))
        torch.testing.assert_close(first_mlp, torch.tensor([[[3.0, 4.0]]]))
        torch.testing.assert_close(second_mlp, torch.tensor([[[3.0, 3.0]]]))
        self.assertEqual(record["target_layers"], [0, 1])
        self.assertEqual(record["module_count"], 2)
        self.assertEqual(record["attention_alpha"], 0.5)
        self.assertEqual(record["mlp_alpha"], 0.25)
        self.assertEqual(record["attention_layers"], [0])
        self.assertEqual(record["mlp_layers"], [1])

    def test_layerwise_weight_hooks_project_an_orthonormal_subspace(self):
        from swift_abliteration.intervention import (
            layerwise_weight_equivalent_ablation_hooks,
        )

        class LinearAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.out_proj = torch.nn.Linear(3, 3, bias=False)

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.down_proj = torch.nn.Linear(3, 3, bias=False)

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_attn = LinearAttention()
                self.mlp = MLP()

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(4, 3)
                self.layers = torch.nn.ModuleList([Layer()])

        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = Backbone()

        cfg = ExperimentConfig(
            name="subspace",
            seed=1,
            model=ModelSpec("fake", "fake", "fake", "fake", 3, 3, 4, 3, 1, 1, 0),
            direction=DirectionSpec(0, 2, "svd", 2, 2, True, "disabled"),
            edit=EditSpec(
                1.0, 0, 0, True, True, False, False, "float32", "bfloat16", True
            ),
        )
        model = Root()
        layer = model.model.language_model.layers[0]
        layer.linear_attn.out_proj.weight.data.copy_(torch.eye(3))
        layer.mlp.down_proj.weight.data.copy_(torch.eye(3))
        basis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        value = torch.tensor([[[3.0, 4.0, 5.0]]])
        with layerwise_weight_equivalent_ablation_hooks(
            model, cfg, {0: basis}
        ) as record:
            observed = layer.linear_attn.out_proj(value)
        torch.testing.assert_close(observed, torch.tensor([[[0.0, 0.0, 5.0]]]))
        self.assertEqual(record["ranks_by_layer"], {"0": 2})

    def test_activation_intervention_can_project_every_target_layer(self):
        from swift_abliteration.intervention import activation_ablation_hooks

        class Add(torch.nn.Module):
            def forward(self, hidden):
                return hidden + torch.tensor([[[1.0, 2.0]]])

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(2, 2)
                self.layers = torch.nn.ModuleList([Add(), Add()])

        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = Backbone()

        model = Root()
        hidden = torch.zeros(1, 1, 2)
        direction = torch.tensor([1.0, 0.0])
        with activation_ablation_hooks(model, direction, [0, 1]):
            for layer in model.model.language_model.layers:
                projected = layer(hidden)
                self.assertEqual(float(projected[0, 0, 0]), 0.0)
                self.assertEqual(float(projected[0, 0, 1]), 2.0)

    def test_reference_activation_ablation_projects_three_sites_per_layer(self):
        from swift_abliteration.intervention import (
            reference_activation_ablation_hooks,
        )

        class Add(torch.nn.Module):
            def __init__(self, delta):
                super().__init__()
                self.register_buffer("delta", torch.tensor(delta))

            def forward(self, hidden):
                return hidden + self.delta

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_attn = Add([[[3.0, 1.0]]])
                self.mlp = Add([[[4.0, 2.0]]])

            def forward(self, hidden_states):
                value = hidden_states + self.linear_attn(hidden_states)
                return value + self.mlp(value)

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(2, 2)
                self.layers = torch.nn.ModuleList([Layer()])

        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = Backbone()

        model = Root()
        layer = model.model.language_model.layers[0]
        hidden = torch.tensor([[[2.0, 2.0]]])
        direction = torch.tensor([1.0, 0.0])
        before = layer(hidden)
        with reference_activation_ablation_hooks(model, direction) as record:
            during = layer(hidden)
        after = layer(hidden)
        self.assertNotEqual(float(before[0, 0, 0]), 0.0)
        self.assertEqual(float(during[0, 0, 0]), 0.0)
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(record["module_count"], 3)
        self.assertFalse(record["weight_equivalent"])

    def test_fake_qwen_edit_removes_direction_from_all_planned_writers(self):
        from swift_abliteration.live_model import apply_edit

        class LinearAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.out_proj = torch.nn.Linear(6, 4, bias=False)

        class FullAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.o_proj = torch.nn.Linear(6, 4, bias=False)

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.down_proj = torch.nn.Linear(8, 4, bias=False)

        class Layer(torch.nn.Module):
            def __init__(self, index):
                super().__init__()
                if index % 4 == 3:
                    self.self_attn = FullAttention()
                else:
                    self.linear_attn = LinearAttention()
                self.mlp = MLP()

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(16, 4)
                self.layers = torch.nn.ModuleList([Layer(i) for i in range(4)])

        class MTP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                layer = torch.nn.Module()
                layer.self_attn = FullAttention()
                layer.mlp = MLP()
                self.layers = torch.nn.ModuleList([layer])

        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = Backbone()
                self.mtp = MTP()

        cfg = ExperimentConfig(
            name="fake",
            seed=1,
            model=ModelSpec(
                id="fake",
                revision="fake",
                architecture="fake",
                model_type="fake",
                hidden_size=4,
                intermediate_size=8,
                vocab_size=16,
                mixer_input_size=6,
                num_layers=4,
                linear_attention_layers=3,
                full_attention_layers=1,
            ),
            direction=DirectionSpec(2, 1, "plain", 2, 2, True, "disabled"),
            edit=EditSpec(
                1.0, 0, 3, True, True, True, True, "float32", "bfloat16", True
            ),
        )
        model = Root()
        direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
        edited = apply_edit(model, cfg, direction)
        self.assertEqual(len(edited), 11)
        self.assertTrue(
            torch.allclose(
                model.model.language_model.embed_tokens.weight @ direction,
                torch.zeros(16),
                atol=1e-6,
            )
        )
        for layer in model.model.language_model.layers:
            mixer = (
                layer.self_attn.o_proj
                if hasattr(layer, "self_attn")
                else layer.linear_attn.out_proj
            )
            self.assertTrue(
                torch.allclose(direction @ mixer.weight, torch.zeros(6), atol=1e-6)
            )
            self.assertTrue(
                torch.allclose(
                    direction @ layer.mlp.down_proj.weight, torch.zeros(8), atol=1e-6
                )
            )


if __name__ == "__main__":
    unittest.main()
