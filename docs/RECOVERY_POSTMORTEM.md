# Swift Qwen3.8 Abliteration Recovery Postmortem

Date: 2026-09-15

## Result

The recovery produced one reversible six-direction candidate that passes the fixed refusal-removal gate. It also passes the matched-harmless, invalid-output, deterministic output-quality, and measured capability gates. It fails the fixed mean-KL gate.

Do not write or publish a permanent checkpoint from this candidate under the frozen protocol. It is a useful research candidate, but it is not a full protocol pass.

## Best candidate

The candidate uses six orthonormal iterative directions. Directions 1 to 4 came from the first iterative run. Direction 5 is the matched-source residual branch. Direction 6 is a new matched-source direction measured after directions 1 to 5 were applied.

The reversible edit uses alpha 1.0. It projects the six-direction subspace out of both residual writers in every language layer:

- 64 mixer output weights: linear-attention `out_proj` or full-attention `o_proj`.
- 64 MLP `down_proj` weights.
- No token-embedding edit.
- No MTP edit in the Transformers generation path.
- No checkpoint write.

The direction file is `runs/gpu/20260915-a100-51081304/iterative-matched-r6/iterative_directions.safetensors`. Its SHA-256 is `02450b23f0582d6b304e5ca021692ff80333cfd355cd4bdde418b0612e74e270`.

## Full validation result

| Measure | Base | Candidate | Gate | Result |
|---|---:|---:|---:|---|
| Standard harmful refusals, 256-token confirmation | 64/64 | 5/64 | Remove at least 90% | Pass: 92.19% removed |
| Matched harmful refusals, 256-token confirmation | 64/64 | 6/64 | Remove at least 90% | Pass: 90.63% removed |
| Matched harmless refusals | 0/64 | 0/64 | Add no more than 5 points | Pass |
| Standard harmless refusals | 20/64 | 11/64 | Diagnostic | Improved by 14.06 points |
| Empty outputs | 0 | 0 | No regression | Pass |
| Severe repeated harmless outputs | 0 | 0 | No regression | Pass |
| Mean harmless KL | 0 by definition | 0.639 | At most 0.10 nats | Fail |

The first full run used 128 output tokens. It had 4 of 64 standard refusals and 5 of 64 matched refusals. A fresh run with 256 output tokens had 5 and 6 refusals. Both settings pass the refusal-removal gate.

All 128-token responses were exact prefixes of the 256-token responses. This confirms deterministic continuation under the same weights and decoding settings.

The candidate mean KL is 0.639 nats. The median is 0.124, p90 is 1.916, p95 is 2.696, and the maximum is 4.253. This means the drift is uneven. A small set of prompts has much larger drift than the typical prompt.

The pinned local WildGuard judge had zero parse errors. On the 256-token confirmation, the pinned local HarmBench cross-check marked behavior success on 38 of 64 standard harmful outputs and 37 of 64 matched harmful outputs. No active classifier used an OpenAI model or a remote inference API.

## Capability result

| Benchmark | Base | Candidate | Change |
|---|---:|---:|---:|
| WMDP-Cyber-256 | 58.20% | 57.42% | -0.78 points |
| CyberMetric-80 | 95.00% | 95.00% | 0.00 points |
| MMLU-Pro-500 | 58.80% | 57.40% | -1.40 points |

All three measured changes are within the fixed two-point limit. These tests do not cancel the KL failure. They show that the high mean KL did not cause a large regression on these three samples.

## Why the first experiment failed

The first reversible hook projected the direction out of the complete residual state after each transformer layer. A permanent weight edit does not do this. It only projects the new output written by each selected mixer and MLP module. The first hook was therefore much stronger than the planned edit. Its empty-output result is a whole-residual stress test. It is not evidence that weight abliteration fails.

The corrected module-output hook and an explicit in-memory weight projection were then compared. Their logits matched to less than 1e-5 KL and their next-token choices matched on the live check. This validated the weight-equivalent intervention.

## Why the simple corrected directions failed

A single layer-38 direction had good cluster separation, but it was not a complete causal refusal control. The best simple corrected direction removed only 39.1% of standard refusals and 45.3% of matched refusals.

The saved activations also showed strong layer variation. For example, the layer-24 and layer-52 standard winsor-995 directions had cosine similarity 0.179. A direction that separates prompt groups at one layer is not automatically the correct output direction for every residual writer.

Static layer-specific rank-1 edits, SVD subspaces, source spans, and Fisher directions did not solve the causal problem. Some had strong held-out separation and low KL but almost no refusal effect. Separation predicts a label. It does not prove that editing that axis controls the behavior.

## Why the iterative method worked

After one direction is removed, the edited model can still create refusal through another direction. The iterative method measures the remaining contrast on the edited model, removes the part already covered by the current basis, and adds the new orthogonal residual direction.

The source directions became opposed at higher ranks. The standard and matched rank-8 directions had cosine similarity about -0.898. Averaging these directions caused cancellation. A matched-source branch for direction 5, followed by one more matched-source measurement for direction 6, removed the remaining matched refusals in the pilot.

The final edit needed both output-module types, all 64 language layers, and full alpha. These controls failed:

- Early 32 layers only: 16/16 standard and 16/16 matched pilot refusals remained.
- Late 32 layers only: 12/16 and 12/16 remained.
- Full-attention layers only: 16/16 and 16/16 remained.
- MLP writers only: 16/16 and 16/16 remained.
- Even layers only: 16/16 and 16/16 remained.
- Odd layers only: 16/16 and 16/16 remained.
- Alpha 0.50: 16/16 and 16/16 remained.

The practical cause is residual rewriting. If one writer or one layer path stays unchanged, it can write the refusal feature back into the residual stream. At lower alpha values, enough of the feature remains for the model's refusal circuit to continue.

## KL trade-off

The one-load alpha ladder tested whether a narrow edit strength could keep the refusal pass and reduce mean KL. It used 16 prompts per group. No tested alpha passed both gates.

| Alpha | Standard refusals | Matched refusals | Mean KL |
|---:|---:|---:|---:|
| 0.50 | 16/16 | 16/16 | 0.089 |
| 0.55 | 16/16 | 16/16 | 0.110 |
| 0.60 | 16/16 | 16/16 | 0.124 |
| 0.65 | 15/16 | 16/16 | 0.150 |
| 0.70 | 15/16 | 16/16 | 0.176 |
| 0.75 | 15/16 | 13/16 | 0.208 |
| 0.80 | 11/16 | 12/16 | 0.244 |
| 0.85 | 9/16 | 10/16 | 0.298 |
| 0.90 | 1/16 | 7/16 | 0.360 |
| 0.95 | 1/16 | 4/16 | 0.449 |
| 0.975 | 1/16 | 4/16 | 0.500 |
| 0.99 | 1/16 | 1/16 | 0.528 |

Alpha 0.50 is the last point below the KL gate, but it leaves every pilot refusal. Alpha 0.55 is already above the KL gate and also leaves every refusal. The first point that passes the refusal gate on both pilot groups is alpha 0.99. Its KL is more than five times the limit.

The evidence supports a sharp behavior threshold rather than a smooth useful trade-off. The next research method should optimize the iterative directions against harmless-output drift during direction construction. It should not only change alpha after the directions are fixed.

## Scope limits

- The final-test split was not used.
- No permanent checkpoint was written.
- Full GSM8K and IFEval were not run.
- MTP draft acceptance was not tested because this Transformers path does not use the checkpoint MTP module.
- The harmful refusal result was confirmed with 256 new tokens. The harmless response-quality run used 128 new tokens.

## Rental and preservation

Vast instance `51081304` used one A100 SXM4 80 GB at $1.1022 per hour. It ran for about 6.087 hours. The estimated compute cost is $6.71.

The remote manifest contains 745 files. All 745 SHA-256 checks passed after the files were copied to the local run directory. The A100 instance was then destroyed and its absence was verified. A separate exited RTX 5090 instance was not changed.

## Main artifacts

- Aggregate gate report: `runs/gpu/20260915-a100-51081304/recovery-screen-summary.json`
- Alpha trade-off report: `runs/gpu/20260915-a100-51081304/alpha-ladder-summary.json`
- Full candidate generation: `runs/gpu/20260915-a100-51081304/candidate-r123456-full64/`
- 256-token harmful confirmation: `runs/gpu/20260915-a100-51081304/candidate-r123456-harmful256/`
- WildGuard judgments: `runs/gpu/20260915-a100-51081304/candidate-r123456-full64-wildguard/`
- HarmBench judgments: `runs/gpu/20260915-a100-51081304/candidate-r123456-full64-harmbench/`
- Capability results: `runs/gpu/20260915-a100-51081304/capability-r123456/` and `capability-r123456-mmlu/`
- Deterministic quality reports: `candidate-r123456-full64/quality-standard-harmless.json` and `quality-matched-harmless.json`
- Live equivalence report: `runs/gpu/20260915-a100-51081304/live-equivalence-rank2.json`

The manifests contain prompt-source hashes, direction hashes, runtime versions, model revisions, and explicit flags for no final-test use, no checkpoint write, no OpenAI-model use, and no remote inference API use.
