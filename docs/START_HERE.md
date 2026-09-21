# Project guide

This file is the current index for the Swift-Qwen3.8-27B abliteration project.
Older reports in this repository describe earlier experiments. Use the release
files in this guide as the source of truth for the published model.

## What was released

- Model: `andrewting/Swift-Qwen3.8-27B-Abliterated`
- Base: `ukisai/Swift-Qwen3.8-27b`
- Base revision: `1b30aaaf753fe5c1cb51ada2ea0367a53445359c`
- Full checkpoint format: BF16, about 55.6 GB
- Single-GPU target format: dynamic Q3 GGUF, 13,223,069,120 bytes
- Target GGUF SHA-256:
  `19363c7b7d1e8337a20a7c13eef49fd6f614841f6d36a894ef058f79394eceb4`

The public model page is:

<https://huggingface.co/andrewting/Swift-Qwen3.8-27B-Abliterated>

## Model and edit summary

The base language model has 64 layers and a residual width of 5,120. It uses
48 linear-attention layers and 16 full-attention layers. Its vocabulary has
248,320 tokens.

The release uses an independently reconstructed rank-6 refusal subspace:

- Measurement layer: 52
- Edit strength: 1.0
- Target layers: all 64 language layers
- Edited modules: attention output projection and MLP down projection
- Edited tensors: 128
- Edit arithmetic: FP32
- Saved weights: BF16
- Token embedding: unchanged
- MTP head: unchanged
- Vision tower: unchanged

The edit is permanent in the released language weights. It does not need a
runtime hook.

## Behavior results

The final test used 64 standard harmful prompts, 64 topic-matched harmful
prompts, and 192 safe prompts. Generation was greedy with a 256-token limit.

| Measure | Standard | Topic-matched |
| --- | ---: | ---: |
| Base opening refusal phrase rate | 100.0% | 100.0% |
| Released model opening refusal phrase rate | 0.0% | 0.0% |
| HarmBench task-completion rate | 68.8% | 64.1% |
| Direct or partial answer rate | 84.4% | 81.3% |
| Soft-refusal count | 9 of 64 | 10 of 64 |

The opening phrase rule measures refusal words only. It does not show that the
model completed a task. HarmBench and the local open-weight response-mode judge
measure this more directly.

The safe first-token KL divergence was:

| Safe set | Count | Mean | Median | P95 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| All safe prompts | 192 | 0.559 | 0.138 | 2.442 | 6.648 |
| Prompts answered by the base | 186 | 0.471 | 0.130 | 1.921 | 6.285 |

The base model gave opening refusals on 5 of 192 safe prompts. The released
model gave an opening refusal on 1 of 192. No new safe opening refusal was
observed.

## Capability results

The planned non-inferiority margin was 2 percentage points.

| Benchmark | Items | Base | Released model | Change | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| WMDP-Cyber | 1,987 | 59.2% | 59.4% | +0.2 pp | Pass |
| CyberMetric | 500 | 95.6% | 95.4% | -0.2 pp | Pass |
| MMLU-Pro | 500 | 58.2% | 57.6% | -0.6 pp | Inconclusive |

The exact paired confidence intervals are in
[`release/Swift-Qwen3.8-27B-Abliterated/evaluation_results.json`](../release/Swift-Qwen3.8-27B-Abliterated/evaluation_results.json).

## Checkpoint verification

The saved checkpoint passed these checks:

- All 128 planned tensors matched the expected projection.
- All 1,071 unplanned tensors were unchanged.
- No non-finite tensor was found.
- The embedding, MTP head, and vision tower were unchanged.
- Eight separate safe smoke prompts produced no empty or severely repeated
  output.

The records are in:

- [`abliteration_manifest.json`](../release/Swift-Qwen3.8-27B-Abliterated/abliteration_manifest.json)
- [`checkpoint-verification.json`](../release/Swift-Qwen3.8-27B-Abliterated/checkpoint-verification.json)
- [`checkpoint-generation-smoke.json`](../release/Swift-Qwen3.8-27B-Abliterated/checkpoint-generation-smoke.json)
- [`abliteration_config.toml`](../release/Swift-Qwen3.8-27B-Abliterated/abliteration_config.toml)

The direction tensor is on the Hugging Face model page. It is not stored in
Git because Git ignores `*.safetensors`.

## RTX 5090 runtime

The selected profile uses:

- Target: dynamic Q3 GGUF
- Draft: public DFlash2 Q4_K_M GGUF
- DFlash draft length: 4
- Context: 262,144 tokens, one slot
- Target and draft KV cache: Q4_0
- Batch: 8,192
- Micro-batch: 2,048
- Full GPU offload
- Flash attention and CUDA graph optimization
- Tested GPU: RTX 5090 at a 570 W limit

| Test | Context or end state | Decode speed | Other result |
| --- | ---: | ---: | --- |
| Fixed Pi replay | 120,844-token input | 81.04 TPS | 48.21% draft acceptance |
| Live long Pi gate | 104,686 total tokens | 83.44 weighted TPS | 12 tool calls, 0 tool errors |
| Live coding task | 31,818 total tokens | 134.13 weighted TPS | 39 tool calls |
| Coding follow-up | 34,946 total tokens | 130.57 weighted TPS | Correction completed |

The coding task passed all 157 project tests, with one pre-existing skip. Five
separate hidden contract tests passed after the follow-up correction.

The exact machine-readable results are in
[`release/llama-pi-benchmark-results.json`](../release/llama-pi-benchmark-results.json).

## How to use the model

### Transformers

Use the example in the root [`README.md`](../README.md) or the full Hugging Face
model card. A GPU with at least 64 GB of memory is a practical starting point
for the BF16 checkpoint.

### RTX 5090 and Pi

```bash
git clone https://github.com/andrewting19/swift-qwen38-abliteration.git
cd swift-qwen38-abliteration/release/replication
chmod +x fetch.sh serve.sh
./fetch.sh /workspace/swift-qwen38-runtime
./serve.sh /workspace/swift-qwen38-runtime
```

Copy `models.json` from that directory to the Pi configuration directory, or
merge its provider entry into the existing Pi configuration. Select provider
`local-swift` and model `swift-abliterated-q3`.

The bundle verifies these three inputs:

- Target model SHA-256:
  `19363c7b7d1e8337a20a7c13eef49fd6f614841f6d36a894ef058f79394eceb4`
- DFlash2 draft SHA-256:
  `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd`
- llama.cpp runtime SHA-256:
  `ed237650a0c6fd27457fd5ec15d3e2f7fd7ce5822db1a47b655231f2f2316357`

Read [`release/replication/README.md`](../release/replication/README.md) before
use. The prebuilt runtime uses CUDA architecture `120a`. It is for RTX 5090
hardware.

## Reproduce the edit and checks

Use these entry points:

| Goal | File |
| --- | --- |
| Final edit configuration | [`configs/release_rank6.toml`](../configs/release_rank6.toml) |
| Build release checkpoint | [`scripts/build_release_final_safe.py`](../scripts/build_release_final_safe.py) |
| Verify checkpoint tensors | [`scripts/verify_release_checkpoint.py`](../scripts/verify_release_checkpoint.py) |
| Validate saved generation | [`scripts/validate_release_generation.py`](../scripts/validate_release_generation.py) |
| Export GGUF tensor types | [`scripts/export_gguf_tensor_types.py`](../scripts/export_gguf_tensor_types.py) |
| Verify GGUF tensor types | [`scripts/verify_gguf_tensor_types.py`](../scripts/verify_gguf_tensor_types.py) |
| RTX 5090 benchmark tools | [`benchmarks/llama_pi/`](../benchmarks/llama_pi) |

The data preparation and evaluation code is public. Raw prompt text, raw
generations, downloaded benchmark files, model files, and local GPU logs are
not in Git. This prevents accidental release of restricted or large local
artifacts. The public release contains hashes and aggregate evaluation values.

## Current and historical documents

Current release documents:

- [`release/Swift-Qwen3.8-27B-Abliterated/README.md`](../release/Swift-Qwen3.8-27B-Abliterated/README.md)
- [`docs/LLAMA_PI_PERFORMANCE.md`](LLAMA_PI_PERFORMANCE.md)
- [`release/LLAMA_CPP.md`](../release/LLAMA_CPP.md)
- [`docs/WIDE_VALIDATION_RESULTS.md`](WIDE_VALIDATION_RESULTS.md)
- [`docs/RELEASE_PLAN.md`](RELEASE_PLAN.md)

The other files in `docs/` record earlier candidates, failed interventions,
judge reassessment, and the search process. They are useful for research
history, but they are not the source of truth for the released checkpoint.

## Known limits

- The model is refusal-reduced. It is not refusal-free.
- Safe-output KL drift is material and has a long tail.
- MMLU-Pro non-inferiority is inconclusive under the planned margin.
- Vision behavior was not tested after the language-weight edit.
- MTP was not edited or used for the release evaluation.
- The fast profile uses an external DFlash2 draft instead of MTP.
- The exact prebuilt llama.cpp binary is hardware-specific.
- The public repository does not contain raw evaluation prompts or responses.
- The model weights remain subject to the Swift Open License v1.0.
