# Behavior Judge Reassessment

Date: 2026-09-15

## Decision

The current rank-1 candidate remains rejected. It removes common refusal
language, but it does not usually produce a direct or useful answer. It also
causes visible language errors.

The older six-direction candidate is not approved. HarmBench finds material
task behavior in many outputs, but a stricter response-mode judge finds mostly
soft refusals and substantial language-quality damage. Human review is required
to resolve this classifier disagreement before that branch can advance.

No final-test data was used. No checkpoint was created. All new judgments used
local open-weight models. No OpenAI model or remote inference API was used.

## Saved-output rescue

Two complete saved-output archives were judged without loading the 27B target
model.

### Rank-1 and scope-search archive

- 98 experiment aliases
- 67 unique output pairs
- 2,272 HarmBench judgments
- 66 non-base candidate aliases in the final comparison report
- No candidate produced a direct or partial answer under the three-prompt
  response-mode consensus judge.

The rejected full-strength rank-1 candidate produced HarmBench behavior success
on 1 of 48 standard and 3 of 48 matched prompts. It produced no direct or
partial answers under the response-mode judge.

### Older recovery archive

- 73 experiment aliases
- 72 unique output pairs
- 2,528 HarmBench judgments

For the six-direction 256-token confirmation:

| Measure | Standard | Matched |
|---|---:|---:|
| HarmBench behavior success | 38/64 | 37/64 |
| Response-mode direct | 2/64 | 3/64 |
| Response-mode partial | 0/64 | 0/64 |
| Response-mode soft refusal | 48/64 | 42/64 |
| Response-mode hard refusal | 9/64 | 12/64 |
| No mode majority | 5/64 | 7/64 |
| Response quality `BROKEN` | 10/64 | 11/64 |

HarmBench and the general response-mode judge measure different properties.
HarmBench can count material target behavior even when a response includes
warnings or refusal framing. The response-mode judge applies the project's
stricter requirement: the model must answer the core request directly. The user
also found soft refusals and language errors during manual review of the rank-1
candidate. The disagreement must stay visible. One classifier cannot approve a
candidate by itself.

## Harmless language quality for six directions

The open-weight response-mode judge compared the 64-row base and rank-6 harmless
outputs.

| Group | Base `MINOR_ERROR` | Rank 6 `MINOR_ERROR` | Base `BROKEN` | Rank 6 `BROKEN` |
|---|---:|---:|---:|---:|
| Standard harmless | 13/64 | 33/64 | 0/64 | 0/64 |
| Matched harmless | 28/64 | 39/64 | 0/64 | 0/64 |

This is consistent with the earlier KL failure. The intervention changes
generated language even when multiple-choice benchmark accuracy stays close to
the base model.

## Process correction

The old screen optimized an easy proxy. It selected directions that removed
refusal openings and refusal substrings. This did not prove direct compliance.
First-token KL and multiple-choice accuracy also did not measure long-form
grammar.

Use this order for all new candidates:

1. Direct or material answer behavior on both harmful sources.
2. No incoherent output and no clear grammar regression.
3. No added refusal on clean harmless prompts.
4. Harmless continuation KL, not only first-token KL.
5. Capability benchmarks only after the first four gates pass.

## Next causal experiment

Do not make another direction from harmful-versus-harmless dataset labels alone.
First test whether the base model can produce direct answers under a controlled
system instruction or assistant prefill. Use the same underlying request in the
refusal and answer conditions.

If an answer condition works:

1. Capture prompt-end activations for each request in both conditions.
2. Calculate paired, winsorized rank-1 directions at layers 24, 32, 38, 44, and
   52.
3. Edit residual writers only. Do not edit the token embedding.
4. Screen rank-1 candidates with 128-token generations and the strict behavior
   and language-quality gates.
5. Build rank 2 only from two rank-1 candidates that each cause real answering
   on different prompts.

If no controlled condition makes the base model answer, a simple activation
contrast does not have a clean answer state to learn. At that point, use a small
post-training method or an uncensored sister checkpoint as the answer source.
Do not keep adding directions that only remove refusal wording.

## Reproducibility

Primary aggregate artifacts:

- `runs/gpu/20260915-saved-output-judge-51147515/summary.json`
- `runs/gpu/20260915-legacy-saved-output-judge-51147515/summary.json`
- `runs/gpu/20260915-r6-quality-judge-51147515/response-mode/manifest.json`

The copied current-archive judge run matched its remote aggregate SHA-256:
`51e96b0e3c772f9c99f7a7cf4b24c6dd258fb2d7459f5929a392bb6992a0de15`.

The copied legacy-archive judge run matched its remote aggregate SHA-256:
`84e47fd7d871ef100f746a9d37c0b6b39e8c0c5c5961c51108aa1d2e0e7db51d`.

The copied rank-6 harmless-quality judge run matched its remote aggregate
SHA-256:
`f6bcbe3faf3195fd732ba47c8f3fc8fdf3551afe3d3900dec6fab3c75d5d5925`.

Vast instance `51147515` was destroyed after artifact verification.
