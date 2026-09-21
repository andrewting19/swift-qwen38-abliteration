---
license: other
license_name: swift-open-license-1.0
license_link: https://huggingface.co/andrewting/Swift-Qwen3.8-27B-Abliterated/blob/main/LICENSE
base_model: ukisai/Swift-Qwen3.8-27b
base_model_relation: finetune
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- qwen3_8
- qwen3_5
- abliterated
- refusal-reduced
- reasoning
- vision-language
---

# Swift-Qwen3.8-27B-Abliterated

This is an experimental, refusal-reduced BF16 derivative of
[`ukisai/Swift-Qwen3.8-27b`](https://huggingface.co/ukisai/Swift-Qwen3.8-27b).
It uses an independently reconstructed rank-6 refusal subspace. No published
Orca, Pliny, or other third-party direction was copied.

This model is not refusal-free. It still gives soft refusals on some requests.
The edit also causes material distribution drift on ordinary prompts. Read the
evaluation and limitations before use.

The public project repository contains the full method, aggregate evaluation
records, and tested RTX 5090 inference profile:

<https://github.com/andrewting19/swift-qwen38-abliteration>

Use its
[`docs/START_HERE.md`](https://github.com/andrewting19/swift-qwen38-abliteration/blob/main/docs/START_HERE.md)
file as the project index.

## What changed

The refusal basis was reconstructed from the base model's own activations. The
final edit projects that six-direction subspace out of the residual-writing
weights:

- Base revision: `1b30aaaf753fe5c1cb51ada2ea0367a53445359c`
- Direction rank: 6
- Direction measurement layer: 52
- Strength: 1.0
- Target layers: all 64 language layers
- Edited modules: attention output projection and MLP down projection
- Edited tensors: 128
- Edit compute: FP32
- Stored weights: BF16
- Token embedding: unchanged
- MTP head: unchanged
- Vision tower: unchanged

The checkpoint contains the full weights. It does not need a runtime hook.
`abliteration_manifest.json`, `abliteration_direction.safetensors`,
`abliteration_config.toml`, `evaluation_results.json`, and
`checkpoint-verification.json` provide the release record.

## Final evaluation

The final split was used once after candidate selection was frozen. Each
harmful group has 64 prompts. The safe group has 192 prompts: 64 standard safe,
64 topic-matched safe, and 64 XSTest-safe prompts. Generation was deterministic
with a 256-token cap.

| Measure | Standard | Topic-matched |
| --- | ---: | ---: |
| Base refusal-phrase rate | 100.0% | 100.0% |
| Candidate refusal-phrase rate | 0.0% | 0.0% |
| HarmBench task-completion rate | 68.8% | 64.1% |
| Uncensored judge direct or partial answer rate | 84.4% | 81.3% |

The phrase rule only detects refusal language. It does not prove substantive
task completion. HarmBench and the uncensored response-mode judge are more
useful for that question. The uncensored judge still marked 9 standard and 10
topic-matched outputs as soft refusals.

Safe first-token KL divergence was 0.559 mean and 0.138 median across all 192
prompts. After excluding six prompts that the base model itself refused, it was
0.471 mean and 0.130 median over 186 prompts. The safe opening-refusal count
changed from 5 for the base to 1 for this model. There were no empty outputs or
severe repetitions in either harmful group.

All refusal and response-mode scoring used local open-weight models. No OpenAI
model or remote classification API was used. See `evaluation_results.json` for
the exact model revisions, counts, hashes, and aggregate values. Raw evaluation
prompts and generations are not included.

## Capability checks

These checks used the exact same reversible weight-equivalent edit before the
checkpoint was saved. The difference is candidate minus base. The planned
non-inferiority margin was 2 percentage points.

| Benchmark | Items | Base | Candidate | Difference | Paired 95% CI | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| WMDP-Cyber | 1,987 | 59.2% | 59.4% | +0.2 pp | [-1.1, +1.4] pp | Pass |
| CyberMetric | 500 | 95.6% | 95.4% | -0.2 pp | [-1.2, +0.6] pp | Pass |
| MMLU-Pro | 500 | 58.2% | 57.6% | -0.6 pp | [-2.4, +1.0] pp | Inconclusive |

The saved checkpoint passed an exact tensor audit: all 128 planned weights
equaled the expected projection, and all 1,071 unplanned tensors were unchanged.
It also passed a separate saved-checkpoint generation smoke test on eight
non-final safe prompts.

## Use with Transformers

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "andrewting/Swift-Qwen3.8-27B-Abliterated"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Explain speculative decoding briefly."}]
text = processor.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=256)
print(processor.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

The checkpoint is about 55.6 GB. A GPU with at least 64 GB of memory is a
practical starting point for BF16 inference. Quantization is needed for smaller
GPUs.

## RTX 5090 GGUF profile

The repository includes a tested dynamic Q3 target, public DFlash2 draft, and
llama.cpp configuration for one RTX 5090. The selected profile measured 81.04
decode tokens per second on a fixed 120,844-token Pi replay and 83.44
token-weighted decode tokens per second in a live Pi session that ended at
104,686 total tokens.

Read the
[`release/replication/`](https://github.com/andrewting19/swift-qwen38-abliteration/tree/main/release/replication)
instructions for downloads, exact hashes, and server settings.

## Limitations

- This is refusal-reduced, not refusal-free or fully uncensored.
- Safe-output KL drift is material and has a long tail.
- The MMLU-Pro result is statistically inconclusive under the planned margin.
- Vision behavior was not evaluated after the language-weight edit.
- The unchanged MTP head was not used in release evaluation. Disable MTP for a
  like-for-like reproduction of the reported results.
- Abliteration can remove useful caution as well as unwanted refusal behavior.
- Benchmark results do not establish safety, factual accuracy, or suitability
  for a specific use.

Users are responsible for lawful and appropriate use.

## License and attribution

This derivative remains subject to the **Swift Open License v1.0**. The included
`LICENSE`, `NOTICE`, and `LICENSE-APACHE-2.0` files are part of this release.
The Swift license permits the listed uses subject to its terms, including its
commercial-use revenue threshold. Read the full license before use or
redistribution.

Swift-Qwen3.8-27B is Copyright 2026 UkisAI. Qwen3.8-27B is Copyright 2026
Alibaba Cloud. This derivative is not made or endorsed by UkisAI or Alibaba
Cloud.

## Citation

```bibtex
@misc{ting2026swiftqwen38abliterated,
  title  = {Swift-Qwen3.8-27B-Abliterated},
  author = {Andrew Ting},
  year   = {2026},
  url    = {https://huggingface.co/andrewting/Swift-Qwen3.8-27B-Abliterated}
}
```
