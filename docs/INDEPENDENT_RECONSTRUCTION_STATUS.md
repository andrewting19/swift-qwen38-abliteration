# Independent Direction Reconstruction Status

Last updated: 2026-09-15

## Objective

Recover a refusal direction from the pinned Swift-Qwen3.8-27B model. Do not use
a published direction as the experiment result. A published edit can be used
only as a positive control that shows that a strong intervention exists.

## What was reconstructed locally

The current direction was calculated from this model's own data and behavior:

1. Build disjoint harmful and harmless candidate pools without using the final
   test split.
2. Generate short base-model responses.
3. Keep harmful cases that the base model refused and harmless cases that the
   base model answered.
4. Select 128 cases from each class.
5. Capture the last-prompt-token residual state at layer 38.
6. Detect the persistent massive-activation coordinate from the captured data.
7. Mask that coordinate and calculate a normalized difference-of-means
   direction.
8. Test the direction with reversible, weight-equivalent residual-writer output
   hooks. No checkpoint was written.

This process independently detected coordinate 3994 as the only massive
activation coordinate. The direction did not copy a published vector.

## Direction quality

- Eligible harmful cases: 159 of 160
- Eligible harmless cases: 158 of 160
- Selected cases: 128 harmful and 128 harmless
- Bootstrap median cosine: 0.997087
- Bootstrap mean cosine: 0.996861
- Cosine with the earlier standard masked direction: 0.950185
- Cosine with the earlier matched masked direction: 0.954641

The direction is measured precisely. Sampling noise at layer 38 is not the main
current problem.

## Causal result

The reversible edit projected this rank-1 direction from the token embedding
and all 64 language-model layers.

| Measure | Standard harmful | Matched harmful |
| --- | ---: | ---: |
| Opening refusals | 4 of 16 | 3 of 16 |
| HarmBench substantive answers | 2 of 16 | 4 of 16 |
| Strict direct or partial answers | 0 of 16 | 1 of 16 |

Safe-screen results:

- Mean harmless continuation KL: 0.035258 nats
- Median harmless continuation KL: 0.025209 nats
- Maximum harmless continuation KL: 0.140374 nats
- Added opening refusals on the two harmless groups and XSTest: 0
- Empty outputs: 0
- Severe repetition: 0

The direction has a real causal effect on refusal openings and has low ordinary
output drift. It does not yet cause reliable substantive answering. Most of the
remaining outputs are soft refusals.

## Interpretation

Statistical direction quality and causal behavior control are different tests.
The layer-38 direction is stable and separates the measured groups, but it
captures only part of the behavior. Increasing the sample count at the same
layer is unlikely to solve this by itself.

The next test is layer choice. Reuse the same 128 plus 128 behavior-filtered
prompts and reconstruct separate directions at layers 24, 32, 38, 44, and 52.
Then compare them with short reversible generation screens. Advance only a
rank-1 direction that increases direct or partial answers. Do not build another
rank-2 direction from directions that only change refusal wording.

## Compute state

Vast instance 51156146 is stopped. The Vast credit balance is zero. Do not
destroy the instance until its remaining judge artifacts are recovered. The
multilayer capture and screen require additional credit or another suitable GPU.

The final-test split remains unused. No permanent edited checkpoint exists.
