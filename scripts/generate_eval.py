#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

from swift_abliteration.gpu_support import (
    read_prompt_jsonl,
    require_acknowledgement,
    require_large_gpu,
)
from swift_abliteration.intervention import activation_ablation_hooks
from swift_abliteration.live_model import (
    capture_last_token_logits,
    render_prompt,
    text_backbone,
)


def parse_layers(value: str) -> list[int]:
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = (int(item) for item in part.split("-", maxsplit=1))
            layers.extend(range(first, last + 1))
        else:
            layers.append(int(part))
    return list(dict.fromkeys(layers))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--logits-output")
    parser.add_argument(
        "--directions", help="Safetensors file made by analyze_directions.py"
    )
    parser.add_argument(
        "--direction-key",
        help="Direction tensor key, for example matched_layer_38_plain",
    )
    parser.add_argument("--intervention-layers")
    parser.add_argument("--intervention-alpha", type=float, default=1.0)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    require_acknowledgement(args.acknowledge)
    require_large_gpu()
    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    prompts, prompt_hash = read_prompt_jsonl(args.prompts)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    tokenizer = getattr(processor, "tokenizer", processor)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = text_backbone(model).embed_tokens.weight.device
    hook_context = nullcontext()
    if bool(args.directions) != bool(args.direction_key):
        raise RuntimeError("Pass both --directions and --direction-key, or neither.")
    if args.directions:
        if not args.intervention_layers:
            raise RuntimeError(
                "Pass --intervention-layers when using a reversible direction."
            )
        from safetensors.torch import load_file

        tensors = load_file(args.directions, device="cpu")
        if args.direction_key not in tensors:
            raise RuntimeError(f"Direction key is not present: {args.direction_key}")
        hook_context = activation_ablation_hooks(
            model,
            tensors[args.direction_key],
            parse_layers(args.intervention_layers),
            args.intervention_alpha,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with hook_context:
        with output.open("w", encoding="utf-8") as handle:
            for index, prompt in enumerate(prompts):
                rendered = render_prompt(tokenizer, prompt, args.system_prompt)
                batch = tokenizer([rendered], return_tensors="pt")
                batch = {name: value.to(device) for name, value in batch.items()}
                with torch.inference_mode():
                    generated = model.generate(
                        **batch,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                    )
                new_tokens = generated[0, batch["input_ids"].shape[1] :]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                record = {"id": index, "prompt": prompt, "response": response}
                line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                handle.write(line)
                digest.update(line.encode())
        if args.logits_output:
            Path(args.logits_output).parent.mkdir(parents=True, exist_ok=True)
            logits = torch.stack(
                capture_last_token_logits(model, processor, prompts, args.system_prompt)
            ).numpy()
            np.savez_compressed(args.logits_output, logits=logits)
    print(
        json.dumps(
            {
                "count": len(prompts),
                "prompt_sha256": prompt_hash,
                "output_sha256": digest.hexdigest(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
