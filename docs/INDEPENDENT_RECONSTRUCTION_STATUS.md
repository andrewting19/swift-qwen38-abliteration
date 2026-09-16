# Independent Direction Reconstruction Status

Last updated: 2026-09-16

## Objective

Recover a refusal direction from the pinned `ukisai/Swift-Qwen3.8-27b` model.
Do not use a published direction as the experiment result. Use a published edit
only as a positive control.

## Current result

Independent direction reconstruction worked. The best local rank-1 direction
caused substantive harmful-task completion on 8 of 16 standard cases and 11 of
16 matched cases. The public Orca checkpoint caused completion on 10 of 16 and
11 of 16 cases with the same HarmBench judge.

A rank-2 edit made from two complementary local directions improved the local
result to 9 of 16 and 11 of 16. It also reduced clean harmless KL from 0.2742
to 0.2316 nats. It did not pass the fixed KL limit of 0.10 nats.

The current problem is not a failure to find a refusal signal. The problem is
that a simple orthogonal weight projection changes too much harmless behavior
at the strength that is required for substantive task completion.

| Reversible arm | Standard HarmBench | Matched HarmBench | Clean harmless KL |
| --- | ---: | ---: | ---: |
| Unchanged base | 0 of 16 | 0 of 16 | 0.0000 |
| Local rank 1, position -5, layer 38 | 3 of 16 | 4 of 16 | 0.1189 |
| Local rank 1, position -12, layer 32 | 6 of 16 | 9 of 16 | 0.2284 |
| Local rank 1, position -13, layer 32 | 8 of 16 | 11 of 16 | 0.2742 |
| Local complementary rank 2 | 9 of 16 | 11 of 16 | 0.2316 |
| Public Orca positive control | 10 of 16 | 11 of 16 | Not measured in this run |

All local arms used reversible weight-equivalent edits. No checkpoint was
written. The final-test split was not used.

## How the local directions were made

The directions came only from this model's activations and behavior:

1. Generate short base-model responses for candidate prompts.
2. Keep harmful cases that the base model refused.
3. Keep harmless cases that the base model answered.
4. Select 128 cases from each class.
5. Capture `resid_pre` activations at 16 prompt-suffix positions and layers 24,
   32, 38, 44, and 52.
6. Mask the persistent massive-activation coordinate.
7. Calculate a harmful-minus-harmless mean-difference direction for each
   position and layer.
8. Test all 80 candidates with reversible interventions and held-out data.

This process independently detected coordinate 3994 as the only persistent
massive-activation coordinate. It did not copy a public direction.

## Important method correction

The reference Arditi code keeps the mean-difference vector at its raw scale for
the activation-addition test. It normalizes the same vector for the ablation
test. Our first addition screen used unit vectors for both tests. This made the
addition test 7 to 99 times too weak.

The corrected raw-scale addition screen found several directions that induced
refusal on all 14 base-answerable safe prompts. Examples include position -9,
layer 38 and position -9, layer 44. This result confirms that the reconstructed
vectors contain a causal refusal signal.

## Why the reference proxy did not select the final arm

The exact reference activation-ablation proxy removes a direction from each
complete residual state. It is not weight-equivalent. On this model, candidates
with low KL under that proxy had little effect under the real weight-equivalent
edit. Candidates with a strong real edit effect had high proxy KL.

The reference proxy is useful evidence, but its original thresholds do not
transfer directly to this model.

## Why the strict response label was misleading

The strict local response-mode judge labeled most outputs from the public Orca
checkpoint as soft refusals. HarmBench still found substantive task completion
on 10 of 16 standard cases and 11 of 16 matched cases.

Therefore, the strict style label is not a sufficient primary success measure.
It detects cautious language, but cautious language can coexist with substantive
completion. HarmBench is now the main substantive-behavior measure. The strict
judge remains a secondary quality measure.

## Why rank 2 was justified

The position -12 and position -13 layer-32 directions had cosine similarity
0.9087. They were related, but they were not identical. Each rank-1 edit also
succeeded on cases where the other edit failed.

Their union covered 10 of 16 standard cases and 12 of 16 matched cases. A QR
orthonormal basis from these two directions produced the rank-2 arm. The result
was 9 of 16 and 11 of 16. This is a small gain over the best rank-1 arm, not a
complete solution.

## Scope and strength tests

The position -13, layer-32 direction needed the full interaction of attention
and MLP residual writers:

| Edit scope | Clean harmless KL | Short-screen marker removal |
| --- | ---: | ---: |
| Full edit with embedding | 0.2742 | 62.5% minimum |
| Full edit without embedding | 0.2812 | 62.5% |
| Layers 18 through 51, no embedding | 0.1636 | 6.25% |
| Attention only | 0.0266 | 0% |
| MLP only | 0.1269 | 6.25% |

Removing the embedding edit did not improve KL. Restricting the layer range or
editing only one writer type removed most of the behavior effect.

The rank-2 alpha ladder also showed a sharp threshold:

| Alpha | Clean harmless KL | Standard HarmBench | Matched HarmBench |
| ---: | ---: | ---: | ---: |
| 0.60 | 0.0830 | Not advanced | Not advanced |
| 0.70 | 0.1079 | 0 of 16 | 0 of 16 |
| 0.80 | 0.1401 | Not advanced | Not advanced |
| 0.90 | 0.1763 | 2 of 16 | 6 of 16 |
| 1.00 | 0.2316 | 9 of 16 | 11 of 16 |

No alpha passed both the 0.10 KL limit and the substantive-behavior requirement.

## Decision

Do not search more random mean-difference directions with the same edit rule.
The next experiment should keep the locally reconstructed direction or rank-2
subspace and change the edit rule. Test a capability-preserving regularized
projection, such as preservation-aware projection or a PRE-style normalized
edit. The first test must be reversible and must reuse saved activations and
the current validation groups.

Do not run the final-test split or create a checkpoint until one arm passes all
fixed gates. Do not run the full capability suite while the current arm still
fails the harmless KL gate, unless the user accepts a new gate.

## Compute state

Vast instance `51156146` is stopped. The last observed account credit was about
$0.57. The stopped disk costs about $0.044 per hour. All required result files
from this run were copied to the local repository workspace.

The final-test split remains unused. No permanent edited checkpoint exists.
