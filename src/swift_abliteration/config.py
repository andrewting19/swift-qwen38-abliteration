from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    id: str
    revision: str
    architecture: str
    model_type: str
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    mixer_input_size: int
    num_layers: int
    linear_attention_layers: int
    full_attention_layers: int


@dataclass(frozen=True)
class DirectionSpec:
    layer: int
    rank: int
    method: str
    harmful_count: int
    harmless_count: int
    last_prompt_token: bool
    massive_activation_mask: str


@dataclass(frozen=True)
class EditSpec:
    alpha: float
    first_layer: int
    last_layer: int
    include_attention_output: bool
    include_mlp_output: bool
    include_embedding: bool
    include_mtp: bool
    edit_compute_dtype: str
    stored_dtype: str
    leave_vision_unchanged: bool


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    model: ModelSpec
    direction: DirectionSpec
    edit: EditSpec


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    cfg = ExperimentConfig(
        name=raw["run"]["name"],
        seed=int(raw["run"]["seed"]),
        model=ModelSpec(**raw["model"]),
        direction=DirectionSpec(**raw["direction"]),
        edit=EditSpec(**raw["edit"]),
    )
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: ExperimentConfig) -> None:
    if cfg.direction.rank != 1:
        raise ValueError("This baseline implements one direction only (rank=1).")
    if not 0 <= cfg.direction.layer < cfg.model.num_layers:
        raise ValueError("Direction layer is outside the text model.")
    if not 0 <= cfg.edit.first_layer <= cfg.edit.last_layer < cfg.model.num_layers:
        raise ValueError("Edit layer range is outside the text model.")
    if not 0.0 <= cfg.edit.alpha <= 1.0:
        raise ValueError("Alpha must be between 0 and 1 for this experiment.")
    if cfg.edit.edit_compute_dtype != "float32":
        raise ValueError("The edit must be calculated in float32.")
    if not cfg.edit.leave_vision_unchanged:
        raise ValueError("This experiment must not edit the vision tower.")
