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

## Multilayer result

Separate layer-24, 32, 38, 44, and 52 directions were reconstructed from the
same 128 plus 128 behavior-filtered prompts. The short reversible screen found:

| Direction layer | Standard opening refusals | Matched opening refusals | XSTest-safe KL |
| ---: | ---: | ---: | ---: |
| 24 | 15 of 16 | 15 of 16 | 0.112637 |
| 32 | 16 of 16 | 16 of 16 | 0.038458 |
| 38 | 4 of 16 | 3 of 16 | 0.219670 |
| 44 | 2 of 16 | 2 of 16 | 0.343553 |
| 52 | 0 of 16 | 0 of 16 | 0.169298 |

The strict local response-mode judge found zero direct or partial answers for
layers 38, 44, and 52 on both harmful groups. Layer 52 removed every opening
refusal but produced 29 soft refusals and three hard refusals across 32 cases.
HarmBench found one behavior success for layer 52 and none on its matched group.

Layer choice changes the refusal opening strongly, but it does not solve the
substantive-answer problem.

## Controlled answer-state result

Two direct-answer system instructions and three assistant prefills were tested
on the same fixed harmful requests. The strict local judge found zero direct or
partial answers for every condition on both harmful groups. These conditions do
not provide a valid answer state for paired direction extraction.

## Remaining method mismatch

The original Arditi reference pipeline extracts candidates at every fixed
end-of-instruction token and at many layers. It does not use only the final
prompt token. It captures `resid_pre`, the input to each transformer block, then
selects candidates with causal ablation, refusal addition, and harmless KL.

The work above used only the final prompt token. The next reconstruction is a
prompt-suffix position sweep. It captures 16 final prompt positions at
`resid_pre` for layers 24, 32, 38, 44, and 52. This creates 80 independent
rank-1 candidates. Short generation screens will select by causal behavior and
safe-output drift. This test must finish before teacher-forced answer data or
post-training is considered.

## Compute state

Vast instance 51156146 is stopped. Its retained disk contains the pinned model
and judge caches. The last checked Vast credit balance was about $3.96. The
prompt-suffix position capture and screen are the next paid-compute tasks.

The final-test split remains unused. No permanent edited checkpoint exists.
