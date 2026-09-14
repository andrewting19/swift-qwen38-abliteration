# Pre-GPU Status

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

## First paid operation

The next step is to rent one GPU, download the 55.6 GB BF16 checkpoint, and run the live architecture and activation capture. No local test can prove the full-model runtime module layout or measure the actual direction.

Do not create a rental until the user confirms the current Vast offer and total hourly price.
