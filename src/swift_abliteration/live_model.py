from __future__ import annotations

from collections.abc import Mapping
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


def apply_runtime_edit(
    model: Any,
    cfg: ExperimentConfig,
    direction: Any,
    alpha: float | None = None,
    *,
    generation_uses_mtp: bool = False,
) -> dict[str, Any]:
    """Apply the configured edit in memory without requiring checkpoint-only MTP."""
    value = cfg.edit.alpha if alpha is None else float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    validate_live_model(model, cfg, require_mtp=False)
    backbone = text_backbone(model)
    edited: list[str] = []
    if cfg.edit.include_embedding:
        output_embeddings = model.get_output_embeddings()
        if (
            output_embeddings is not None
            and output_embeddings.weight is backbone.embed_tokens.weight
        ):
            raise RuntimeError(
                "The input embedding and output head are tied. The configured edit "
                "does not include the output head."
            )
        project_embedding_rows_(backbone.embed_tokens.weight, direction, value)
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
            project_output_weight_(module.weight, direction, value)
            edited.append(f"model.language_model.layers.{index}.{label}.weight")
        if cfg.edit.include_mlp_output:
            project_output_weight_(layer.mlp.down_proj.weight, direction, value)
            edited.append(f"model.language_model.layers.{index}.mlp.down_proj.weight")
    if cfg.edit.include_mtp and generation_uses_mtp:
        mtp_layer = mtp_root(model).layers[0]
        if cfg.edit.include_attention_output:
            project_output_weight_(mtp_layer.self_attn.o_proj.weight, direction, value)
            edited.append("mtp.layers.0.self_attn.o_proj.weight")
        if cfg.edit.include_mlp_output:
            project_output_weight_(mtp_layer.mlp.down_proj.weight, direction, value)
            edited.append("mtp.layers.0.mlp.down_proj.weight")
    return {
        "type": "in_memory_weight_projection",
        "edited_tensor_count": len(edited),
        "edited_tensors": edited,
        "embedding_included": cfg.edit.include_embedding,
        "mtp_included": bool(cfg.edit.include_mtp and generation_uses_mtp),
        "mtp_exclusion_reason": None
        if not cfg.edit.include_mtp or generation_uses_mtp
        else "The active Transformers generation path does not execute MTP.",
        "alpha": value,
        "checkpoint_saved": False,
    }


def apply_layerwise_runtime_edit(
    model: Any,
    cfg: ExperimentConfig,
    directions_by_layer: Mapping[int, Any],
    alpha: float = 1.0,
    *,
    embedding_direction: Any | None = None,
) -> dict[str, Any]:
    """Apply one nonpersistent layerwise weight projection in memory."""
    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    validate_live_model(model, cfg, require_mtp=False)
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

    edited: list[str] = []
    if embedding_direction is not None:
        output_embeddings = model.get_output_embeddings()
        if (
            output_embeddings is not None
            and output_embeddings.weight is backbone.embed_tokens.weight
        ):
            raise RuntimeError(
                "The input embedding and output head are tied. The edit does not "
                "include the output head."
            )
        project_embedding_rows_(
            backbone.embed_tokens.weight, embedding_direction, value
        )
        edited.append("model.language_model.embed_tokens.weight")

    for index, direction in sorted(assignments.items()):
        layer = backbone.layers[index]
        if cfg.edit.include_attention_output:
            if hasattr(layer, "linear_attn"):
                module = layer.linear_attn.out_proj
                label = "linear_attn.out_proj"
            else:
                module = layer.self_attn.o_proj
                label = "self_attn.o_proj"
            project_output_weight_(module.weight, direction, value)
            edited.append(f"model.language_model.layers.{index}.{label}.weight")
        if cfg.edit.include_mlp_output:
            project_output_weight_(layer.mlp.down_proj.weight, direction, value)
            edited.append(f"model.language_model.layers.{index}.mlp.down_proj.weight")

    return {
        "type": "layerwise_in_memory_weight_projection",
        "edited_tensor_count": len(edited),
        "edited_tensors": edited,
        "target_layers": sorted(assignments),
        "ranks_by_layer": {
            str(index): 1 if direction.ndim == 1 else int(direction.shape[0])
            for index, direction in sorted(assignments.items())
        },
        "embedding_included": embedding_direction is not None,
        "mtp_included": False,
        "alpha": value,
        "checkpoint_saved": False,
    }


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
    batch_size: int = 1,
) -> list[Any]:
    torch = __import__("torch")
    backbone = text_backbone(model)
    tokenizer = getattr(processor, "tokenizer", processor)
    results: list[Any] = []
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            values = prompts[start : start + batch_size]
            texts = [render_prompt(tokenizer, value, system_prompt) for value in values]
            batch = tokenizer(texts, return_tensors="pt", padding=True)
            device = backbone.embed_tokens.weight.device
            batch = {
                name: value.to(device)
                for name, value in batch.items()
                if hasattr(value, "to")
            }
            with torch.inference_mode():
                output = model(**batch, use_cache=False)
            results.extend(output.logits[:, -1].float().cpu().unbind(0))
    finally:
        tokenizer.padding_side = previous_padding_side
    return results


def generate_responses_with_first_logits(
    model: Any,
    processor: Any,
    prompts: list[str],
    max_new_tokens: int,
    system_prompt: str | None = None,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Generate deterministic responses and retain only first-step logits."""
    torch = __import__("torch")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    backbone = text_backbone(model)
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError("Tokenizer has neither a pad token nor an EOS token.")
    responses: list[str] = []
    first_step_logits: list[Any] = []
    first_token_ids: list[int] = []
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            values = prompts[start : start + batch_size]
            texts = [render_prompt(tokenizer, value, system_prompt) for value in values]
            batch = tokenizer(texts, return_tensors="pt", padding=True)
            device = backbone.embed_tokens.weight.device
            batch = {
                name: value.to(device)
                for name, value in batch.items()
                if hasattr(value, "to")
            }
            with torch.inference_mode():
                generated = model.generate(
                    **batch,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True,
                    pad_token_id=pad_token_id,
                )
            if not generated.scores:
                raise RuntimeError("Generation did not return first-step logits.")
            input_width = int(batch["input_ids"].shape[1])
            first_step_logits.extend(generated.scores[0].float().cpu().unbind(0))
            first_token_ids.extend(
                int(value) for value in generated.sequences[:, input_width].cpu()
            )
            for token_ids in generated.sequences[:, input_width:]:
                responses.append(tokenizer.decode(token_ids, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = previous_padding_side
    if not (
        len(responses) == len(first_step_logits) == len(first_token_ids) == len(prompts)
    ):
        raise RuntimeError("Generated response and logit counts do not match prompts.")
    return {
        "responses": responses,
        "first_step_logits": first_step_logits,
        "first_token_ids": first_token_ids,
    }


def capture_prompt_and_first_output_activations_multi(
    model: Any,
    processor: Any,
    prompts: list[str],
    layer_indices: list[int],
    system_prompt: str | None = None,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Capture prompt-end and first-generated-token states in one cached generation."""
    torch = __import__("torch")
    backbone = text_backbone(model)
    tokenizer = getattr(processor, "tokenizer", processor)
    indices = tuple(dict.fromkeys(int(index) for index in layer_indices))
    if not indices:
        raise ValueError("At least one capture layer is required.")
    invalid = [index for index in indices if not 0 <= index < len(backbone.layers)]
    if invalid:
        raise ValueError(f"Capture layers are outside the model: {invalid}")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    calls: dict[int, list[Any]] = {index: [] for index in indices}

    def make_hook(index: int):
        def hook(_module: Any, _inputs: Any, output: Any):
            tensor = output[0] if isinstance(output, tuple) else output
            calls[index].append(tensor.detach())

        return hook

    handles = [
        backbone.layers[index].register_forward_hook(make_hook(index))
        for index in indices
    ]
    activations = {
        "prompt_end": {index: [] for index in indices},
        "first_output": {index: [] for index in indices},
    }
    first_step_logits: list[Any] = []
    first_token_ids: list[int] = []
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            values = prompts[start : start + batch_size]
            texts = [render_prompt(tokenizer, value, system_prompt) for value in values]
            batch = tokenizer(texts, return_tensors="pt", padding=True)
            device = backbone.embed_tokens.weight.device
            batch = {
                name: value.to(device)
                for name, value in batch.items()
                if hasattr(value, "to")
            }
            for captured in calls.values():
                captured.clear()
            with torch.inference_mode():
                generated = model.generate(
                    **batch,
                    do_sample=False,
                    max_new_tokens=2,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True,
                )
            if len(generated.scores) < 1:
                raise RuntimeError("Generation did not return first-step logits.")
            input_width = int(batch["input_ids"].shape[1])
            first_step_logits.extend(generated.scores[0].float().cpu().unbind(0))
            first_token_ids.extend(
                int(value) for value in generated.sequences[:, input_width].cpu()
            )
            for index in indices:
                if len(calls[index]) < 2:
                    raise RuntimeError(
                        "Two generation forwards are required to capture the first "
                        f"output token at layer {index}; observed {len(calls[index])}."
                    )
                prompt_values = calls[index][0][:, -1].float().cpu()
                first_output_values = calls[index][1][:, -1].float().cpu()
                activations["prompt_end"][index].extend(prompt_values.unbind(0))
                activations["first_output"][index].extend(first_output_values.unbind(0))
    finally:
        tokenizer.padding_side = previous_padding_side
        for handle in handles:
            handle.remove()
    return {
        "activations": activations,
        "first_step_logits": first_step_logits,
        "first_token_ids": first_token_ids,
    }
