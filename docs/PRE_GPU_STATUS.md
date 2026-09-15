# Pre-GPU Status

The authoritative project state and restart instructions are in `docs/PROJECT_PLAN.md`.

## Completed

1. Research public dataset recipes without printing harmful prompt text.
2. Pin two direction datasets and two disjoint holdout datasets.
3. Add a semantic-matched contrast set to reduce topic confounding.
4. Define a five-layer and four-estimator direction study.
5. Implement reversible activation projection and permanent FP32 weight projection.
6. Freeze refusal, over-refusal, KL, coherence, general, cyber, and runtime evaluations with pass thresholds. Define MTP acceptance as a later compatibility test.
7. Pin the Python packages, benchmark revisions, GPU constraints, disk plan, cost ceiling, logging, hashing, and cleanup procedure.

## Verified locally

- Public model metadata selects 131 tensors for Arm A and 68 tensors for Arm B.
- Transformers resolves the pinned model to `Qwen3_5ForConditionalGeneration`.
- The processor and chat template load without model weights.
- The chat template runs with thinking disabled and a fixed system prompt.
- The permanent edit streams the original safetensors shards. This preserves checkpoint-only MTP tensors that the Transformers runtime ignores.
- MTP draft acceptance is not part of the first paid run because the pinned Transformers runtime does not load the MTP module.
- Generated prompt and benchmark files match their pinned SHA-256 hashes.
- Unit tests cover direction math, winsorization, bootstrap stability, layer intervention, weight projection, KL, and judge parsing.

## Final controls completed

- Separate candidate-validation and final-test splits are pinned and disjoint.
- The OpenAI-compatible reversible benchmark server has local API tests.
- The initial judge path used `gpt-5-nano` with minimal reasoning and JSON output. A harmless API check passed. Live results later showed low confidence and poor cross-judge agreement, so the recorded evaluation amendment uses repeated `gpt-5.6-luna` high passes with xhigh tie-breaks.
- The Vast account reported about $18.49 credit before rental. The user authorized use of the available balance.
- The Linux GPU environment uses PyTorch 2.9.1 with CUDA 12.8, `causal-conv1d` 1.7.0, and `flash-linear-attention` 0.5.2. This avoids the slow reference implementation used when the optimized Qwen kernels are absent.

## First paid operation

The next step is to rent one GPU, download the 55.6 GB BF16 checkpoint, and run the live architecture and activation capture. No local test can prove the full-model runtime module layout or measure the actual direction.

The local controls pass. Search current Vast offers immediately before rental. Keep enough credit and time to copy and verify completed artifacts before shutdown.
