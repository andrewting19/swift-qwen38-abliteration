from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the GPU dependencies with: pip install -e '.[gpu]'") from exc
    return torch


def calculate_direction(harmful: Iterable[Any], harmless: Iterable[Any]):
    torch = require_torch()
    h = torch.stack([item.detach().float().cpu() for item in harmful])
    s = torch.stack([item.detach().float().cpu() for item in harmless])
    if h.ndim != 2 or s.ndim != 2 or h.shape[1] != s.shape[1]:
        raise ValueError("Activation groups must have shape [prompts, hidden_size].")
    raw = h.mean(0) - s.mean(0)
    norm = torch.linalg.vector_norm(raw)
    if not torch.isfinite(norm) or norm.item() <= 0:
        raise ValueError("The measured direction has zero or invalid length.")
    return raw / norm


def project_output_weight_(weight: Any, direction: Any, alpha: float, column_chunk: int = 1024) -> None:
    """Edit a [hidden, input] tensor with bounded FP32 temporary memory."""
    torch = require_torch()
    r = direction.detach().to(device=weight.device, dtype=torch.float32)
    r = r / torch.linalg.vector_norm(r)
    if weight.ndim != 2 or weight.shape[0] != r.numel():
        raise ValueError("Output weight shape does not match the direction.")
    with torch.no_grad():
        for start in range(0, weight.shape[1], column_chunk):
            stop = min(start + column_chunk, weight.shape[1])
            block = weight[:, start:stop].float()
            block.sub_(r[:, None] * (r @ block)[None, :], alpha=alpha)
            weight[:, start:stop].copy_(block.to(dtype=weight.dtype))


def project_embedding_rows_(weight: Any, direction: Any, alpha: float, row_chunk: int = 1024) -> None:
    """Edit a [vocab, hidden] tensor with bounded FP32 temporary memory."""
    torch = require_torch()
    r = direction.detach().to(device=weight.device, dtype=torch.float32)
    r = r / torch.linalg.vector_norm(r)
    if weight.ndim != 2 or weight.shape[1] != r.numel():
        raise ValueError("Embedding shape does not match the direction.")
    with torch.no_grad():
        for start in range(0, weight.shape[0], row_chunk):
            stop = min(start + row_chunk, weight.shape[0])
            block = weight[start:stop].float()
            block.sub_((block @ r)[:, None] * r[None, :], alpha=alpha)
            weight[start:stop].copy_(block.to(dtype=weight.dtype))


def save_direction(direction: Any, path: str | Path, metadata: dict[str, str]) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("Install safetensors with the GPU dependency set.") from exc
    tensor = direction.detach().float().cpu().contiguous()
    save_file({"refusal_direction": tensor}, str(path), metadata=metadata)
