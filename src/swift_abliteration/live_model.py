from __future__ import annotations

from typing import Any

from .config import ExperimentConfig
from .torch_ops import project_embedding_rows_, project_output_weight_


def text_backbone(model: Any) -> Any:
    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise RuntimeError("Cannot find the Qwen text language_model.layers module.")


def mtp_root(model: Any) -> Any:
    candidates = [
        getattr(model, "mtp", None),
        getattr(getattr(model, "model", None), "mtp", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise RuntimeError(
        "The plan includes MTP, but the live model has no mtp.layers module."
    )


def validate_live_model(
    model: Any, cfg: ExperimentConfig, require_mtp: bool = False
) -> None:
    backbone = text_backbone(model)
    if len(backbone.layers) != cfg.model.num_layers:
        raise RuntimeError(
            f"Expected {cfg.model.num_layers} text layers, found {len(backbone.layers)}."
        )
    expected_embedding = (cfg.model.vocab_size, cfg.model.hidden_size)
    if tuple(backbone.embed_tokens.weight.shape) != expected_embedding:
        raise RuntimeError(
            f"Expected embedding shape {expected_embedding}, found "
            f"{tuple(backbone.embed_tokens.weight.shape)}."
        )
    for index, layer in enumerate(backbone.layers):
        down = layer.mlp.down_proj.weight
        if tuple(down.shape) != (cfg.model.hidden_size, cfg.model.intermediate_size):
            raise RuntimeError(
                f"Unexpected MLP output shape at layer {index}: {tuple(down.shape)}"
            )
        if index % 4 == 3:
            mixer = getattr(layer, "self_attn", None)
            projection = getattr(mixer, "o_proj", None)
            expected_name = "self_attn.o_proj"
        else:
            mixer = getattr(layer, "linear_attn", None)
            projection = getattr(mixer, "out_proj", None)
            expected_name = "linear_attn.out_proj"
        if projection is None:
            raise RuntimeError(f"Expected {expected_name} at layer {index}.")
        expected_mixer = (cfg.model.hidden_size, cfg.model.mixer_input_size)
        if tuple(projection.weight.shape) != expected_mixer:
            raise RuntimeError(
                f"Expected {expected_name} shape {expected_mixer} at layer {index}, "
                f"found {tuple(projection.weight.shape)}."
            )
    if cfg.edit.include_mtp and require_mtp:
        mtp_layer = mtp_root(model).layers[0]
        expected_mixer = (cfg.model.hidden_size, cfg.model.mixer_input_size)
        expected_down = (cfg.model.hidden_size, cfg.model.intermediate_size)
        if tuple(mtp_layer.self_attn.o_proj.weight.shape) != expected_mixer:
            raise RuntimeError(
                "MTP attention output shape does not match the frozen architecture."
            )
        if tuple(mtp_layer.mlp.down_proj.weight.shape) != expected_down:
            raise RuntimeError(
                "MTP MLP output shape does not match the frozen architecture."
            )


def apply_edit(model: Any, cfg: ExperimentConfig, direction: Any) -> list[str]:
    validate_live_model(model, cfg, require_mtp=cfg.edit.include_mtp)
    backbone = text_backbone(model)
    edited: list[str] = []
    if cfg.edit.include_embedding:
        project_embedding_rows_(backbone.embed_tokens.weight, direction, cfg.edit.alpha)
        edited.append("model.language_model.embed_tokens.weight")
    for index in range(cfg.edit.first_layer, cfg.edit.last_layer + 1):
        layer = backbone.layers[index]
        if cfg.edit.include_attention_output:
            if hasattr(layer, "linear_attn"):
                module = layer.linear_attn.out_proj
                label = "linear_attn.out_proj"
            else:
                module = layer.self_attn.o_proj
                label = "self_attn.o_proj"
            project_output_weight_(module.weight, direction, cfg.edit.alpha)
            edited.append(f"model.language_model.layers.{index}.{label}.weight")
        if cfg.edit.include_mlp_output:
            project_output_weight_(
                layer.mlp.down_proj.weight, direction, cfg.edit.alpha
            )
            edited.append(f"model.language_model.layers.{index}.mlp.down_proj.weight")
    if cfg.edit.include_mtp:
        mtp_layer = mtp_root(model).layers[0]
        if cfg.edit.include_attention_output:
            project_output_weight_(
                mtp_layer.self_attn.o_proj.weight, direction, cfg.edit.alpha
            )
            edited.append("mtp.layers.0.self_attn.o_proj.weight")
        if cfg.edit.include_mlp_output:
            project_output_weight_(
                mtp_layer.mlp.down_proj.weight, direction, cfg.edit.alpha
            )
            edited.append("mtp.layers.0.mlp.down_proj.weight")
    return edited


def render_prompt(tokenizer: Any, prompt: str, system_prompt: str | None = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def capture_last_token_activations(
    model: Any,
    processor: Any,
    prompts: list[str],
    layer_index: int,
    system_prompt: str | None = None,
) -> list[Any]:
    return capture_last_token_activations_multi(
        model, processor, prompts, [layer_index], system_prompt
    )[layer_index]


def capture_last_token_activations_multi(
    model: Any,
    processor: Any,
    prompts: list[str],
    layer_indices: list[int],
    system_prompt: str | None = None,
) -> dict[int, list[Any]]:
    torch = __import__("torch")
    backbone = text_backbone(model)
    indices = tuple(dict.fromkeys(int(index) for index in layer_indices))
    if not indices:
        raise ValueError("At least one capture layer is required.")
    invalid = [index for index in indices if not 0 <= index < len(backbone.layers)]
    if invalid:
        raise ValueError(f"Capture layers are outside the model: {invalid}")
    captured: dict[int, list[Any]] = {index: [] for index in indices}

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[index].append(tensor.detach())

        return hook

    handles = [
        backbone.layers[index].register_forward_hook(make_hook(index))
        for index in indices
    ]
    results: dict[int, list[Any]] = {index: [] for index in indices}
    try:
        for prompt in prompts:
            tokenizer = getattr(processor, "tokenizer", processor)
            text = render_prompt(tokenizer, prompt, system_prompt)
            batch = tokenizer([text], return_tensors="pt", padding=True)
            device = backbone.embed_tokens.weight.device
            batch = {
                name: value.to(device)
                for name, value in batch.items()
                if hasattr(value, "to")
            }
            for values in captured.values():
                values.clear()
            with torch.inference_mode():
                model(**batch, use_cache=False)
            position = int(batch["attention_mask"][0].sum().item()) - 1
            for index in indices:
                if len(captured[index]) != 1:
                    raise RuntimeError(
                        f"Expected one activation at layer {index}, captured {len(captured[index])}."
                    )
                results[index].append(captured[index][0][0, position].float().cpu())
    finally:
        for handle in handles:
            handle.remove()
    return results


def capture_last_token_logits(
    model: Any,
    processor: Any,
    prompts: list[str],
    system_prompt: str | None = None,
) -> list[Any]:
    torch = __import__("torch")
    backbone = text_backbone(model)
    tokenizer = getattr(processor, "tokenizer", processor)
    results: list[Any] = []
    for prompt in prompts:
        text = render_prompt(tokenizer, prompt, system_prompt)
        batch = tokenizer([text], return_tensors="pt", padding=True)
        device = backbone.embed_tokens.weight.device
        batch = {
            name: value.to(device)
            for name, value in batch.items()
            if hasattr(value, "to")
        }
        with torch.inference_mode():
            output = model(**batch, use_cache=False)
        position = int(batch["attention_mask"][0].sum().item()) - 1
        results.append(output.logits[0, position].float().cpu())
    return results
