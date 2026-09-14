from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any
from urllib.request import urlopen

from .config import ExperimentConfig


class ArchitectureMismatch(RuntimeError):
    pass


def hf_json(model_id: str, revision: str, filename: str) -> dict[str, Any]:
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def validate_public_metadata(
    cfg: ExperimentConfig,
    model_config: dict[str, Any],
    weight_index: dict[str, Any],
) -> dict[str, Any]:
    text = model_config.get("text_config", model_config)
    checks = {
        "architecture": model_config.get("architectures") == [cfg.model.architecture],
        "model_type": model_config.get("model_type") == cfg.model.model_type,
        "hidden_size": text.get("hidden_size") == cfg.model.hidden_size,
        "intermediate_size": text.get("intermediate_size")
        == cfg.model.intermediate_size,
        "vocab_size": text.get("vocab_size") == cfg.model.vocab_size,
        "num_layers": text.get("num_hidden_layers") == cfg.model.num_layers,
    }
    layer_types = text.get("layer_types", [])
    checks["layer_type_count"] = len(layer_types) == cfg.model.num_layers
    checks["linear_attention_count"] = (
        layer_types.count("linear_attention") == cfg.model.linear_attention_layers
    )
    checks["full_attention_count"] = (
        layer_types.count("full_attention") == cfg.model.full_attention_layers
    )
    expected_pattern = [
        "full_attention" if index % 4 == 3 else "linear_attention"
        for index in range(cfg.model.num_layers)
    ]
    checks["layer_pattern"] = layer_types == expected_pattern
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ArchitectureMismatch("Metadata checks failed: " + ", ".join(failed))

    weight_map = weight_index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ArchitectureMismatch("The checkpoint has no safetensors weight map.")
    selected = select_weight_names(cfg, layer_types, set(weight_map))
    expected_count = expected_writer_count(cfg)
    if len(selected) != expected_count:
        raise ArchitectureMismatch(
            f"Expected {expected_count} editable tensors, found {len(selected)}."
        )
    if any("visual" in name or "vision" in name for name in selected):
        raise ArchitectureMismatch("The edit plan selected a vision tensor.")

    return {
        "experiment": cfg.name,
        "model": asdict(cfg.model),
        "checks": checks,
        "planned_tensor_count": len(selected),
        "planned_tensors": selected,
        "checkpoint_bytes": weight_index.get("metadata", {}).get("total_size"),
    }


def expected_writer_count(cfg: ExperimentConfig) -> int:
    layers = cfg.edit.last_layer - cfg.edit.first_layer + 1
    per_layer = int(cfg.edit.include_attention_output) + int(
        cfg.edit.include_mlp_output
    )
    total = layers * per_layer
    if cfg.edit.include_embedding:
        total += 1
    if cfg.edit.include_mtp:
        total += int(cfg.edit.include_attention_output) + int(
            cfg.edit.include_mlp_output
        )
    return total


def select_weight_names(
    cfg: ExperimentConfig,
    layer_types: list[str],
    available: set[str],
) -> list[str]:
    names: list[str] = []
    if cfg.edit.include_embedding:
        names.append("model.language_model.embed_tokens.weight")
    for layer in range(cfg.edit.first_layer, cfg.edit.last_layer + 1):
        prefix = f"model.language_model.layers.{layer}"
        if cfg.edit.include_attention_output:
            mixer = (
                "linear_attn.out_proj"
                if layer_types[layer] == "linear_attention"
                else "self_attn.o_proj"
            )
            names.append(f"{prefix}.{mixer}.weight")
        if cfg.edit.include_mlp_output:
            names.append(f"{prefix}.mlp.down_proj.weight")
    if cfg.edit.include_mtp:
        if cfg.edit.include_attention_output:
            names.append("mtp.layers.0.self_attn.o_proj.weight")
        if cfg.edit.include_mlp_output:
            names.append("mtp.layers.0.mlp.down_proj.weight")
    missing = [name for name in names if name not in available]
    if missing:
        raise ArchitectureMismatch("Missing planned tensors: " + ", ".join(missing))
    return names
