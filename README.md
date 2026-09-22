# Swift-Qwen3.8-27B-Abliterated

This repository contains the code, aggregate evaluation records, and tested
inference profile for
[`andrewting/Swift-Qwen3.8-27B-Abliterated`](https://huggingface.co/andrewting/Swift-Qwen3.8-27B-Abliterated).

The model is an experimental, refusal-reduced derivative of
[`ukisai/Swift-Qwen3.8-27b`](https://huggingface.co/ukisai/Swift-Qwen3.8-27b).
It uses a refusal direction that was reconstructed from the base model. It does
not copy a published Orca, Pliny, or other third-party direction.

Start with [docs/START_HERE.md](docs/START_HERE.md) for the full file map,
methods, results, limits, and reproduction paths.

## Main results

The released checkpoint uses a rank-6 refusal subspace measured at language
layer 52. The edit removes this subspace from the attention output and MLP down
projection in all 64 language layers. It changes 128 weight tensors. It does
not change the token embedding, MTP head, or vision tower.

| Measure | Result |
| --- | ---: |
| Opening refusal phrase rate, standard / topic-matched | 0.0% / 0.0% |
| HarmBench task completion, standard / topic-matched | 68.8% / 64.1% |
| Direct or partial answer, standard / topic-matched | 84.4% / 81.3% |
| Safe first-token KL, mean / median, 192 prompts | 0.559 / 0.138 |
| WMDP-Cyber change | +0.2 percentage points |
| CyberMetric change | -0.2 percentage points |
| MMLU-Pro change | -0.6 percentage points, inconclusive |

The model is not refusal-free. It gave soft refusals on 9 of 64 standard and
10 of 64 topic-matched final-test prompts. Safe-output distribution drift is
material. See the [model card](release/Swift-Qwen3.8-27B-Abliterated/README.md)
and [aggregate results](release/Swift-Qwen3.8-27B-Abliterated/evaluation_results.json).

## Available weights

- Full BF16 checkpoint: about 55.6 GB on
  [Hugging Face](https://huggingface.co/andrewting/Swift-Qwen3.8-27B-Abliterated)
- RTX 5090 target:
  `Swift-Qwen3.8-27B-Abliterated-UD-Q3_K_XL.gguf`, 13.22 GB
- Fast speculative draft:
  `incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`

## Use the BF16 checkpoint

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
print(
    processor.tokenizer.decode(
        output[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
)
```

A GPU with at least 64 GB of memory is a practical starting point for BF16
inference. Use a quantized file for a smaller GPU.

## Reproduce the RTX 5090 profile

The public bundle downloads the exact model, DFlash2 draft, and tested
SM120a llama.cpp runtime. It verifies all SHA-256 values before use.

```bash
git clone https://github.com/andrewting19/swift-qwen38-abliteration.git
cd swift-qwen38-abliteration/release/replication
chmod +x fetch.sh serve.sh
./fetch.sh /workspace/swift-qwen38-runtime
./serve.sh /workspace/swift-qwen38-runtime
```

The selected RTX 5090 profile measured:

- 81.04 decode tokens per second on a fixed 120,844-token Pi replay
- 83.44 token-weighted decode tokens per second in a live Pi session that
  ended at 104,686 total tokens
- 134.13 token-weighted decode tokens per second in a 31,818-token coding task

The fixed replay had 48.21% draft acceptance. Hardware, drivers, prompt
content, and draft acceptance can change these results.

See [release/replication/README.md](release/replication/README.md) for the exact
host requirements and [docs/LLAMA_PI_PERFORMANCE.md](docs/LLAMA_PI_PERFORMANCE.md)
for all measured runtime results.

## Repository map

| Path | Purpose |
| --- | --- |
| `docs/START_HERE.md` | Full project summary and file map |
| `docs/MIMO_V26_CRACK_FORENSICS.md` | Public third-party CRACK checkpoint comparison |
| `release/Swift-Qwen3.8-27B-Abliterated/` | Model card, edit record, aggregate results, and verification records |
| `release/replication/` | Exact RTX 5090 download and server bundle |
| `benchmarks/llama_pi/` | Pi replay and live-agent benchmark tools |
| `src/swift_abliteration/` | Direction analysis and weight-edit library |
| `scripts/` | Data, direction, screening, evaluation, release, and verification tools |
| `configs/release_rank6.toml` | Final model edit configuration |
| `docs/` | Current reports and historical experiment records |

Raw evaluation prompts, raw generations, model files, credentials, and local
GPU logs are excluded from Git. The public release contains aggregate results
and hashes only.

## Limits

- Abliteration can remove useful caution and can damage normal model behavior.
- Vision behavior was not evaluated after the language-weight edit.
- The unchanged MTP head was not used in the release evaluation.
- The fast llama.cpp profile uses an external DFlash2 draft. It does not use
  the model's MTP head.
- The public prebuilt llama.cpp runtime targets RTX 5090 hardware.
- Benchmark results do not establish safety, factual accuracy, or fitness for
  a specific use.

The model weights remain subject to the Swift Open License v1.0. Read the
license in the Hugging Face release before use or redistribution.
