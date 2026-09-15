import sys
import unittest
import copy
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
    def test_in_memory_weight_projection_supports_row_subspaces(self):
        from swift_abliteration.torch_ops import (
            project_embedding_rows_,
            project_output_weight_,
        )

        basis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        output_weight = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        )
        embedding = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
        project_output_weight_(output_weight, basis, 1.0, column_chunk=1)
        project_embedding_rows_(embedding, basis, 1.0, row_chunk=1)
        torch.testing.assert_close(output_weight[:2], torch.zeros(2, 2))
        torch.testing.assert_close(
            embedding[:, :2], torch.zeros(2, 2)
        )
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
            edit=EditSpec(1.0, 0, 0, True, True, True, False, "float32", "bfloat16", True),
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
                hooked.model.language_model.layers[0].linear_attn.out_proj(attention_input),
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
            edit=EditSpec(1.0, 0, 1, True, True, False, False, "float32", "bfloat16", True),
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
            model, cfg, directions
        ) as record:
            first = model.model.language_model.layers[0].linear_attn.out_proj(value)
            second = model.model.language_model.layers[1].linear_attn.out_proj(value)
        torch.testing.assert_close(first, torch.tensor([[[0.0, 4.0]]]))
        torch.testing.assert_close(second, torch.tensor([[[3.0, 0.0]]]))
        self.assertEqual(record["target_layers"], [0, 1])
        self.assertEqual(record["module_count"], 4)

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
            edit=EditSpec(1.0, 0, 0, True, True, False, False, "float32", "bfloat16", True),
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
