from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig
from .live_model import mtp_root, text_backbone


@dataclass(frozen=True)
class RuntimeWriter:
    name: str
    module: Any
    kind: str


def project_activation(tensor: Any, direction: Any, alpha: float = 1.0) -> Any:
    """Remove alpha times one direction or an orthonormal row subspace."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    value = direction.to(device=tensor.device, dtype=tensor.dtype)
    if value.ndim == 1:
        unit = value / value.norm().clamp_min(1e-12)
        return tensor - alpha * (tensor @ unit).unsqueeze(-1) * unit
    if value.ndim == 2:
        if value.shape[0] == 0:
            raise ValueError("A direction subspace must have at least one row.")
        return tensor - alpha * ((tensor @ value.transpose(0, 1)) @ value)
    raise ValueError("Direction must be a vector or an orthonormal row matrix.")


@contextmanager
def activation_addition_input_hook(
    model: Any,
    direction: Any,
    layer_index: int,
    coefficient: float = 1.0,
) -> Iterator[dict[str, Any]]:
    """Temporarily add one direction to every token at one layer input."""
    backbone = text_backbone(model)
    index = int(layer_index)
    if not 0 <= index < len(backbone.layers):
        raise ValueError(f"Activation-addition layer is outside the model: {index}")
    value = float(coefficient)
    if not value:
        raise ValueError("Activation-addition coefficient must be nonzero.")

    def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if args:
            hidden = args[0]
            vector = direction.to(device=hidden.device, dtype=hidden.dtype)
            return (hidden + value * vector, *args[1:]), kwargs
        if "hidden_states" not in kwargs:
            raise RuntimeError("The layer input has no hidden-state tensor.")
        updated = dict(kwargs)
        hidden = updated["hidden_states"]
        vector = direction.to(device=hidden.device, dtype=hidden.dtype)
        updated["hidden_states"] = hidden + value * vector
        return args, updated

    handle = backbone.layers[index].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield {
            "type": "activation_addition_at_layer_input",
            "layer": index,
            "coefficient": value,
            "checkpoint_saved": False,
        }
    finally:
        handle.remove()


def _replace_hidden(output: Any, direction: Any, alpha: float) -> Any:
    if isinstance(output, tuple):
        return (project_activation(output[0], direction, alpha), *output[1:])
    if isinstance(output, list):
        return [project_activation(output[0], direction, alpha), *output[1:]]
    return project_activation(output, direction, alpha)


def _project_linear_output(
    output: Any, direction: Any, alpha: float, bias: Any | None
) -> Any:
    """Match a projected linear weight while leaving its bias unchanged."""
    if bias is None:
        return project_activation(output, direction, alpha)
    adjusted_bias = bias.to(device=output.device, dtype=output.dtype)
    return project_activation(output - adjusted_bias, direction, alpha) + adjusted_bias


def planned_runtime_writers(
    model: Any,
    cfg: ExperimentConfig,
    *,
    generation_uses_mtp: bool = False,
) -> list[RuntimeWriter]:
    """Return modules whose outputs match the planned checkpoint weight edits."""
    backbone = text_backbone(model)
    targets: list[RuntimeWriter] = []
    if cfg.edit.include_embedding:
        output_embeddings = model.get_output_embeddings()
        if (
            output_embeddings is not None
            and output_embeddings.weight is backbone.embed_tokens.weight
        ):
            raise RuntimeError(
                "The input embedding and output head are tied. An embedding-output "
                "hook would not match the configured checkpoint edit."
            )
        targets.append(
            RuntimeWriter(
                "model.language_model.embed_tokens", backbone.embed_tokens, "embedding"
            )
        )
    for index in range(cfg.edit.first_layer, cfg.edit.last_layer + 1):
        layer = backbone.layers[index]
        if cfg.edit.include_attention_output:
            if hasattr(layer, "linear_attn"):
                module = layer.linear_attn.out_proj
                label = "linear_attn.out_proj"
            else:
                module = layer.self_attn.o_proj
                label = "self_attn.o_proj"
            targets.append(
                RuntimeWriter(
                    f"model.language_model.layers.{index}.{label}", module, "linear"
                )
            )
        if cfg.edit.include_mlp_output:
            targets.append(
                RuntimeWriter(
                    f"model.language_model.layers.{index}.mlp.down_proj",
                    layer.mlp.down_proj,
                    "linear",
                )
            )
    if cfg.edit.include_mtp and generation_uses_mtp:
        mtp_layer = mtp_root(model).layers[0]
        if cfg.edit.include_attention_output:
            targets.append(
                RuntimeWriter(
                    "mtp.layers.0.self_attn.o_proj",
                    mtp_layer.self_attn.o_proj,
                    "linear",
                )
            )
        if cfg.edit.include_mlp_output:
            targets.append(
                RuntimeWriter(
                    "mtp.layers.0.mlp.down_proj", mtp_layer.mlp.down_proj, "linear"
                )
            )
    return targets


@contextmanager
def weight_equivalent_ablation_hooks(
    model: Any,
    cfg: ExperimentConfig,
    direction: Any,
    alpha: float | None = None,
    *,
    generation_uses_mtp: bool = False,
) -> Iterator[dict[str, Any]]:
    """Temporarily match the configured output-weight and embedding projections."""
    value = cfg.edit.alpha if alpha is None else float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    targets = planned_runtime_writers(
        model, cfg, generation_uses_mtp=generation_uses_mtp
    )
    if not targets:
        raise ValueError("The configured edit contains no runtime writer modules.")
    hidden_size = int(direction.shape[-1])
    handles = []
    bias_modules: list[str] = []
    try:
        for target in targets:
            if target.kind == "embedding":
                if (
                    target.module.weight.ndim != 2
                    or target.module.weight.shape[1] != hidden_size
                ):
                    raise ValueError(
                        f"Embedding shape does not match direction: {target.name}"
                    )

                def hook(
                    _module,
                    _inputs,
                    output,
                    *,
                    _direction=direction,
                    _alpha=value,
                ):
                    return project_activation(output, _direction, _alpha)
            else:
                if (
                    target.module.weight.ndim != 2
                    or target.module.weight.shape[0] != hidden_size
                ):
                    raise ValueError(
                        f"Writer shape does not match direction: {target.name}"
                    )
                bias = getattr(target.module, "bias", None)
                if bias is not None:
                    bias_modules.append(target.name)

                def hook(
                    _module,
                    _inputs,
                    output,
                    *,
                    _direction=direction,
                    _alpha=value,
                    _bias=bias,
                ):
                    return _project_linear_output(output, _direction, _alpha, _bias)

            handles.append(target.module.register_forward_hook(hook))
        yield {
            "type": "weight_equivalent_module_output_projection",
            "module_count": len(targets),
            "modules": [target.name for target in targets],
            "bias_modules": bias_modules,
            "embedding_included": cfg.edit.include_embedding,
            "mtp_included": bool(cfg.edit.include_mtp and generation_uses_mtp),
            "mtp_exclusion_reason": None
            if not cfg.edit.include_mtp or generation_uses_mtp
            else "The active Transformers generation path does not execute MTP.",
            "alpha": value,
        }
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def layerwise_weight_equivalent_ablation_hooks(
    model: Any,
    cfg: ExperimentConfig,
    directions_by_layer: Mapping[int, Any],
    alpha: float = 1.0,
    *,
    embedding_direction: Any | None = None,
    attention_alpha: float | None = None,
    mlp_alpha: float | None = None,
    embedding_alpha: float | None = None,
    attention_layers: set[int] | None = None,
    mlp_layers: set[int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Project each selected layer's writers along its assigned direction."""
    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    attention_value = value if attention_alpha is None else float(attention_alpha)
    mlp_value = value if mlp_alpha is None else float(mlp_alpha)
    embedding_value = value if embedding_alpha is None else float(embedding_alpha)
    for label, component_value in (
        ("attention_alpha", attention_value),
        ("mlp_alpha", mlp_value),
        ("embedding_alpha", embedding_value),
    ):
        if not 0.0 <= component_value <= 1.0:
            raise ValueError(f"{label} must be between 0 and 1")
    backbone = text_backbone(model)
    assignments = {
        int(index): direction for index, direction in directions_by_layer.items()
    }
    if not assignments:
        raise ValueError("At least one layer direction is required.")
    invalid = [
        index
        for index in assignments
        if not cfg.edit.first_layer <= index <= cfg.edit.last_layer
        or not 0 <= index < len(backbone.layers)
    ]
    if invalid:
        raise ValueError(
            f"Layerwise intervention layers are outside the edit: {invalid}"
        )
    attention_layer_set = (
        set(assignments) if attention_layers is None else set(attention_layers)
    )
    mlp_layer_set = set(assignments) if mlp_layers is None else set(mlp_layers)
    invalid_components = sorted(
        (attention_layer_set | mlp_layer_set).difference(assignments)
    )
    if invalid_components:
        raise ValueError(
            f"Component layer sets are outside the direction assignments: {invalid_components}"
        )

    handles = []
    module_names: list[str] = []
    bias_modules: list[str] = []

    def register_linear(
        name: str, module: Any, direction: Any, projection_alpha: float
    ) -> None:
        hidden_size = int(direction.shape[-1])
        if module.weight.ndim != 2 or module.weight.shape[0] != hidden_size:
            raise ValueError(f"Writer shape does not match direction: {name}")
        bias = getattr(module, "bias", None)
        if bias is not None:
            bias_modules.append(name)
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, *, _direction=direction, _bias=bias, _alpha=projection_alpha: (
                    _project_linear_output(output, _direction, _alpha, _bias)
                )
            )
        )
        module_names.append(name)

    try:
        if embedding_direction is not None:
            hidden_size = int(embedding_direction.shape[-1])
            if (
                backbone.embed_tokens.weight.ndim != 2
                or backbone.embed_tokens.weight.shape[1] != hidden_size
            ):
                raise ValueError("Embedding shape does not match its direction.")
            handles.append(
                backbone.embed_tokens.register_forward_hook(
                    lambda _module, _inputs, output, *, _direction=embedding_direction: (
                        project_activation(output, _direction, embedding_value)
                    )
                )
            )
            module_names.append("model.language_model.embed_tokens")

        for index, direction in sorted(assignments.items()):
            layer = backbone.layers[index]
            if cfg.edit.include_attention_output and index in attention_layer_set:
                if hasattr(layer, "linear_attn"):
                    module = layer.linear_attn.out_proj
                    label = "linear_attn.out_proj"
                else:
                    module = layer.self_attn.o_proj
                    label = "self_attn.o_proj"
                register_linear(
                    f"model.language_model.layers.{index}.{label}",
                    module,
                    direction,
                    attention_value,
                )
            if cfg.edit.include_mlp_output and index in mlp_layer_set:
                register_linear(
                    f"model.language_model.layers.{index}.mlp.down_proj",
                    layer.mlp.down_proj,
                    direction,
                    mlp_value,
                )
        yield {
            "type": "layerwise_weight_equivalent_module_output_projection",
            "module_count": len(module_names),
            "modules": module_names,
            "bias_modules": bias_modules,
            "target_layers": sorted(assignments),
            "ranks_by_layer": {
                str(index): 1 if direction.ndim == 1 else int(direction.shape[0])
                for index, direction in sorted(assignments.items())
            },
            "embedding_included": embedding_direction is not None,
            "mtp_included": False,
            "alpha": value,
            "attention_alpha": attention_value,
            "mlp_alpha": mlp_value,
            "embedding_alpha": embedding_value
            if embedding_direction is not None
            else None,
            "attention_layers": sorted(attention_layer_set),
            "mlp_layers": sorted(mlp_layer_set),
        }
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def activation_ablation_hooks(
    model: Any,
    direction: Any,
    layer_indices: Sequence[int],
    alpha: float = 1.0,
) -> Iterator[None]:
    """Temporarily project a direction from selected transformer-layer outputs."""
    backbone = text_backbone(model)
    indices = tuple(dict.fromkeys(int(index) for index in layer_indices))
    if not indices:
        raise ValueError("At least one intervention layer is required.")
    invalid = [index for index in indices if not 0 <= index < len(backbone.layers)]
    if invalid:
        raise ValueError(f"Intervention layers are outside the model: {invalid}")

    handles = []
    try:
        for index in indices:
            handles.append(
                backbone.layers[index].register_forward_hook(
                    lambda _module, _inputs, output, *, _direction=direction, _alpha=alpha: (
                        _replace_hidden(output, _direction, _alpha)
                    )
                )
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def reference_activation_ablation_hooks(
    model: Any,
    direction: Any,
    layer_indices: Sequence[int] | None = None,
    alpha: float = 1.0,
) -> Iterator[dict[str, Any]]:
    """Reproduce the Arditi candidate-selection activation ablation proxy.

    This is not weight-equivalent. It projects the direction from every selected
    block input and from each selected attention and MLP output. Use it only as
    a causal direction-selection proxy, then validate the selected direction
    with the reversible weight-equivalent intervention.
    """
    backbone = text_backbone(model)
    indices = tuple(
        range(len(backbone.layers))
        if layer_indices is None
        else dict.fromkeys(int(index) for index in layer_indices)
    )
    if not indices:
        raise ValueError("At least one intervention layer is required.")
    invalid = [index for index in indices if not 0 <= index < len(backbone.layers)]
    if invalid:
        raise ValueError(f"Intervention layers are outside the model: {invalid}")

    handles = []
    module_names: list[str] = []

    def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if args:
            return (project_activation(args[0], direction, alpha), *args[1:]), kwargs
        if "hidden_states" not in kwargs:
            raise RuntimeError("The layer input has no hidden-state tensor.")
        updated = dict(kwargs)
        updated["hidden_states"] = project_activation(
            updated["hidden_states"], direction, alpha
        )
        return args, updated

    def output_hook(_module: Any, _inputs: Any, output: Any):
        return _replace_hidden(output, direction, alpha)

    try:
        for index in indices:
            layer = backbone.layers[index]
            handles.append(
                layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
            )
            module_names.append(f"model.language_model.layers.{index}.resid_pre")
            if hasattr(layer, "linear_attn"):
                attention = layer.linear_attn
                attention_name = "linear_attn"
            else:
                attention = layer.self_attn
                attention_name = "self_attn"
            handles.append(attention.register_forward_hook(output_hook))
            handles.append(layer.mlp.register_forward_hook(output_hook))
            module_names.extend(
                [
                    f"model.language_model.layers.{index}.{attention_name}",
                    f"model.language_model.layers.{index}.mlp",
                ]
            )
        yield {
            "type": "reference_activation_ablation_proxy",
            "target_layers": list(indices),
            "module_count": len(module_names),
            "modules": module_names,
            "alpha": float(alpha),
            "weight_equivalent": False,
            "checkpoint_saved": False,
        }
    finally:
        for handle in handles:
            handle.remove()
