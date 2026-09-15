# Swift Qwen3.8 Abliteration Experiment Report

Date: 2026-09-15

## Recovery addendum

This report records the first rental. A later recovery run found a six-direction candidate that passed the fixed refusal-removal gate on both harmful validation groups. It also passed the matched-harmless, invalid-output, deterministic quality, and three measured capability gates. It failed the fixed mean-KL gate at 0.639 nats. The current decision is still to not create a permanent checkpoint. See `docs/RECOVERY_POSTMORTEM.md` for the current result.

## Initial-run decision

Do not create a permanent checkpoint from the tested directions.

The two nondestructive weight-equivalent candidates changed refusal behavior, but neither came close to the fixed requirement. The best result removed 39.1% of base refusals on the standard harmful set and 45.3% on the matched harmful set. The requirement was at least 90% on both sets.

The plain matched direction was destructive. It produced empty decoded output for all 256 validation items.

## Active evaluation policy

All active classification ran locally on the rented A100 with open-weight models. No active classification used an OpenAI model or a remote inference API.

- Primary refusal and over-refusal judge: `Kotovskiy/Wildguard-Qwen3-4b`, revision `e2a675e253900267ffc9e7c7aebe4c987ac9de0f`.
- Harmful-behavior cross-check: `cais/HarmBench-Llama-2-13b-cls`, revision `bda705349d1144fa618770bea64d99ce54e3835b`.
- WildGuard classifications: 768, with zero parse errors.
- HarmBench classifications: 384, with zero parse errors.

Earlier OpenAI-model judgment artifacts are discontinued diagnostics. They were not used for this decision.

## Corrected reversible screen

| Arm | Refusal removal, standard | Refusal removal, matched | HarmBench success, standard | HarmBench success, matched | Mean harmless KL | Empty outputs | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Consensus layer-38 winsor-995 | 37.5% | 40.6% | 10.9% | 10.9% | 0.033 | 0/256 | Reject |
| Standard layer-38 winsor-995 | 39.1% | 45.3% | 9.4% | 15.6% | 0.027 | 0/256 | Reject |
| Matched layer-38 plain | Not scored | Not scored | Not scored | Not scored | 15.106 | 256/256 | Reject as destructive |

The base WildGuard refusal count was 64 of 64 on each harmful set. The base HarmBench behavior-success count was 0 of 64 on each harmful set.

On the matched harmless set, the base and both winsor-995 candidates had zero refusals. On the standard harmless set, the consensus arm reduced the refusal rate by 3.1 percentage points. The standard arm increased it by 1.6 percentage points.

## Interpretation

The direction-quality statistics were strong, but they did not predict a strong enough causal refusal change. The standard winsor-995 direction produced the largest refusal reduction. Its effect was still less than half of the required effect on both validation sources.

The low KL values show that both winsor-995 edits made a small average change to the model's ordinary last-token distribution. They do not show that the edits removed refusal. The refusal labels show that the intervention was too weak or that one direction was not sufficient.

The plain matched direction shows the opposite failure. A direction can have good separation and bootstrap stability but still contain high-impact model features. Removing it from all residual writers can destroy generation.

## Work not run

- The final-test split was not used.
- The quick capability benchmarks were not run because no candidate passed the refusal screen.
- No permanent checkpoint was written.
- No MTP draft-acceptance test was run.
- No OpenAI-model classifications will be used again in this project.

## Reproducibility

The intervention equivalence check passed on 129 runtime residual writers. Batch-4 greedy generation matched batch-1 generation on the fixed pilot.

All 81 files in the remote run directory were copied to the local run directory. Remote and local SHA-256 hashes matched for every file before instance destruction. The local rental record was then updated with the final end time and cost. The aggregate report is `runs/gpu/20260914-a100-51065040/metrics/open_weight_screen_summary.json`. It contains no raw prompt or response text.

The rental ran for 3.894 hours at $1.1022 per hour. Its estimated cost was $4.29. Vast instance `51065040` was destroyed after transfer verification.

The first whole-transformer-layer-output experiment remains labeled as a stress test. It is not a valid proxy for the checkpoint weight edit and did not affect this decision.
