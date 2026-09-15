# Abliteration Recovery Plan

Date: 2026-09-15

## Goal

Find a weight-equivalent intervention that removes at least 90% of base refusals on both frozen harmful validation sets without failing the harmless-refusal, KL, or invalid-output gates. Use only local open-weight classifiers. Preserve a postmortem even if no candidate passes.

## Root-cause hypothesis

The first corrected screen used one layer-38 direction for the embedding and all 128 residual-writing modules. This assumes that the same refusal axis is valid across the complete 64-layer network.

The saved activations do not support that assumption. For standard winsor-995 directions, cross-layer cosine similarity ranges from 0.179 to 0.715. The layer-24 and layer-52 directions are almost different axes. Matched directions show the same pattern. A layer-38 direction can separate prompts well at layer 38 but still be the wrong output axis for most other layers.

The plain matched direction failed for a different reason. It produced 256 empty outputs and 15.106 mean KL. Its high separation and bootstrap stability did not protect it from high-impact activation outliers or useful-feature overlap.

## Recovery sequence

1. Use the five saved direction depths: 24, 32, 38, 44, and 52.
2. Build a normalized standard-plus-matched consensus direction at each depth.
3. Assign each edited layer the direction from its nearest measured depth.
4. Compare exact-anchor, radius-2, late-layer, and middle-layer target ranges.
5. Run a 16-item-per-group pilot with 128 output tokens.
6. Score the pilot with local WildGuard only. Reject empty-output and high-KL arms.
7. Run the strongest plausible recipes on all 64 validation items per group with 256 output tokens.
8. Confirm passing candidates with local WildGuard and HarmBench.
9. Run quick capability benchmarks only after a candidate passes the refusal screen.
10. Use final-test data and write a permanent checkpoint only after every validation gate passes.

## Execution findings

- Eight layer-specific rank-1 arms produced 576 non-empty outputs. All mean harmless KL values were below 0.10. None removed more than one of 32 harmful refusals.
- Three rank-1 controls also produced only non-empty outputs. The best control was a global layer-52 consensus direction. It removed 3 of 16 standard and 6 of 16 matched refusals, with mean KL 0.084. It did not reach the 90% gate.
- Six no-embedding rank-2 and rank-4 subspace arms completed. A simple refusal-phrase check found refusals in every harmful output. The incomplete rank-8 arm was stopped after 12 records because low-rank hook overhead made it too slow and the lower-rank arms had no early effect.
- The layer-specific and subspace pilots did not project the token embedding output. This leaves a refusal component that later residual-writer projections cannot remove. The next pilots add an explicit embedding direction. The intervention remains exactly equivalent to projecting the embedding weight rows and the selected residual-writer output weights.

## Controls

- No OpenAI model or remote inference API is permitted for classification.
- Existing OpenAI judgment artifacts remain discontinued diagnostics.
- The first whole-residual intervention remains a stress test only.
- Every new intervention projects only embedding or residual-writer module output and is weight-equivalent.
- No raw harmful prompt or generation text goes into Git, the dashboard, or chat.
- Use one paid GPU at a time.

## Recovery rental

- Vast instance: `51081304`
- GPU: A100 SXM4 80 GB
- Hourly rate: $1.1022
- Local run directory: `runs/gpu/20260915-a100-51081304`
