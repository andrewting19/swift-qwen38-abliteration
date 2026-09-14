from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from .live_model import text_backbone


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
