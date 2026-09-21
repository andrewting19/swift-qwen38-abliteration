# RTX 5090 llama.cpp and Pi performance

Date: 2026-09-19

## Selected runtime

- Target: `Swift-Qwen3.8-27B-Abliterated-UD-Q3_K_XL.gguf`
- Target SHA-256: `19363c7b7d1e8337a20a7c13eef49fd6f614841f6d36a894ef058f79394eceb4`
- Target size: 13,223,069,120 bytes
- Draft: `incoai/Qwen3.8-27B-DFlash2-GGUF`, Q4_K_M
- Draft SHA-256: `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd`
- DFlash `n_max`: 4
- Context: 262,144 tokens, one slot
- Target and draft KV: Q4_0
- Batch and micro-batch: 8,192 and 2,048
- CUDA graph optimization: enabled
- GPU offload: all layers
- Hardware: RTX 5090 at a 570 W limit, AMD EPYC 7542, Vast instance 51635634

The target uses the exact 866-tensor layout of the reference dynamic Q3
package. The verification checked all 506 quantized tensor types. It did not
copy reference model weights.

## Fixed 120,844-token Pi replay

| Draft configuration | Decode TPS | Draft acceptance | Result |
|---|---:|---:|---|
| Q4_0, `n_max=6` | 71.94 | 29.52% | Reject |
| Q4_0, `n_max=4` | 76.41 | 43.67% | Reject |
| Q4_K_M, `n_max=4` | **81.04** | **48.21%** | Accept |

The more accurate Q4_K_M draft improved target agreement enough to offset its
small extra compute cost.

## Live Pi agent tests

The main coding task used 32 agent turns and 39 tool calls. It generated 11,207
tokens at a token-weighted **134.13 TPS** and reached 31,818 total context
tokens. A follow-up correction reached 34,946 total context tokens and ran at
**130.57 TPS**.

The final long-context gate resumed a saved real Pi session. It ended at
104,686 total tokens, used 12 shell-tool calls with no tool errors, and
generated 4,863 tokens at a token-weighted **83.44 TPS**. It completed the
review task and stopped normally.

The coding task passed all 157 project tests, with one pre-existing skip. Five
separate hidden contract tests also passed after the follow-up correction.

## Preserved evidence

Logs, Pi event streams, diffs, metrics, hashes, and summaries are under:

`runs/gpu/20260919-llama-pi-51635634/`

Use `benchmarks/llama_pi/run_llama_server.sh dflash` for the selected server
profile. The script defaults to Q4_K_M and `n_max=4`.
