from __future__ import annotations

from typing import Any

from .torch_ops import project_embedding_rows_, project_output_weight_


def edit_shard_tensors(
    tensors: dict[str, Any],
    planned: set[str],
    direction: Any,
    alpha: float,
) -> list[str]:
    edited = []
    for name in sorted(planned & set(tensors)):
        if name == "model.language_model.embed_tokens.weight":
            project_embedding_rows_(tensors[name], direction, alpha)
        else:
            project_output_weight_(tensors[name], direction, alpha)
        tensors[name] = tensors[name].contiguous()
        edited.append(name)
    return edited
