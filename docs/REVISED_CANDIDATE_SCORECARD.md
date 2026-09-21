# Revised Candidate Scorecard

Date: 2026-09-18

## Decision

The local iterative rank-6 edit is the lead wider-validation candidate. The local complementary rank-2 edit is the best low-rank alternative. No candidate is final.

KL is now a drift diagnostic. It is not an automatic rejection rule. HarmBench substantive completion is the primary effectiveness measure. Refusal-language removal is secondary.

| Candidate | Standard completion | Matched completion | KL mean / median | Output gate | Capability | Decision |
| --- | ---: | ---: | ---: | --- | --- | --- |
| local iterative rank 6 | 38/64 (59.4%) | 37/64 (57.8%) | 0.639 / 0.124 | pass | partly inconclusive | promote to wider validation |
| local complementary rank 2 | 9/16 (56.2%) | 11/16 (68.8%) | 0.232 / 0.165 | pass | not run | promising low rank pilot |
| local rank 1, position -13, layer 32 | 8/16 (50.0%) | 11/16 (68.8%) | 0.274 / 0.176 | pass | not run | superseded by complementary rank 2 |
| local rank 1, position -12, layer 32 | 6/16 (37.5%) | 9/16 (56.2%) | 0.228 / 0.154 | pass | not run | superseded by complementary rank 2 |
| public Swift uncensored positive control | 10/16 (62.5%) | 11/16 (68.8%) | not measured | pass | not measured | reference only |

## Rank-6 interpretation

The rank-6 edit removed refusal language from 59 of 64 standard cases and 58 of 64 matched cases. Actual HarmBench completion was lower: 38 of 64 and 37 of 64. Thus, refusal-language removal must not be called task completion.

On the first 16 cases, the rank-6 edit completed 11 standard and 13 matched tasks. The public control completed 10 and 11. The paired intervals are wide, so this is evidence of comparable behavior, not proof that rank 6 is better.

The rank-6 capability point changes were small. Paired 95% intervals are:

- WMDP-Cyber-256: -0.8%; 95% CI -3.9% to +2.3%; inconclusive.
- CyberMetric-80: +0.0%; 95% CI +0.0% to +0.0%; pass.
- MMLU-Pro-500: -1.4%; 95% CI -3.0% to +0.2%; inconclusive.

WMDP-Cyber and MMLU-Pro are inconclusive under a strict 2-point noninferiority margin. CyberMetric passes because every paired correctness result was unchanged.

## Required wider validation

1. Run all 250 XSTest safe prompts and classify full, partial, and false refusal.
2. Run at least 100 held-out harmful requests for the public control under the same token limit, then compare it with rank 6.
3. Run the missing GSM8K and IFEval tests. Expand WMDP-Cyber and MMLU-Pro until their paired intervals can support a decision.
4. Measure token-level KL or teacher-forced cross-entropy on 500 to 2,000 clean prompts. Report median, p90, p95, and outliers.
5. Manually audit fixed samples and the largest-drift safe outputs before one candidate is frozen.
6. Keep the final-test split unused until the candidate and rules are frozen.

## Data limits

The rank-2 and rank-1 results use only 16 prompts per harmful source and 16 safe prompts. Their intervals are wide. They are pilot results only. The rank-6 safe test did not use the complete XSTest-250 set.

No final-test data was used. No checkpoint was written.
