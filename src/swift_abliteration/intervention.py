from __future__ import annotations

from collections.abc import Iterator, Sequence
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
    """Remove alpha times the component along direction from the last axis."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    unit = direction.to(device=tensor.device, dtype=tensor.dtype)
    unit = unit / unit.norm().clamp_min(1e-12)
    return tensor - alpha * (tensor @ unit).unsqueeze(-1) * unit


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
                RuntimeWriter("mtp.layers.0.mlp.down_proj", mtp_layer.mlp.down_proj, "linear")
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
    hidden_size = int(direction.numel())
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
                hook = (
                    lambda _module,
                    _inputs,
                    output,
                    *,
                    _direction=direction,
                    _alpha=value: project_activation(output, _direction, _alpha)
                )
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
                hook = (
                    lambda _module,
                    _inputs,
                    output,
                    *,
                    _direction=direction,
                    _alpha=value,
                    _bias=bias: _project_linear_output(
                        output, _direction, _alpha, _bias
                    )
                )
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
