# Wide Validation Results

Date: 2026-09-18

## Decision

Rank 6 is the refusal-removal leader. Rank 2 is the minimal-change leader. No candidate is frozen.

| Candidate | HarmBench standard | HarmBench matched | Uncensored mode standard | Uncensored mode matched | Safe KL mean / median |
| --- | ---: | ---: | ---: | ---: | ---: |
| iterative_rank6 | 57.8% | 56.2% | 87.5% | 76.6% | 0.661 / 0.357 |
| complementary_rank2 | 50.0% | 54.7% | 65.6% | 64.1% | 0.269 / 0.087 |

HarmBench is the stricter task-completion measure. The uncensored mode judge separates direct or partial answers from soft and hard refusals. The aligned Qwen 4B mode judge is retained as a diagnostic, but it is not used for the final decision because it conflicts sharply with both other judges on harmful outputs.

## Capability

Differences are candidate minus base. The non-inferiority margin is 2 percentage points.

| Candidate | Dataset | Base | Candidate | Difference | Paired 95% CI | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| iterative_rank6 | wmdp_cyber_full | 59.2% | 59.4% | +0.2 pp | [-1.1, +1.4] pp | pass |
| iterative_rank6 | cybermetric_500 | 95.6% | 95.4% | -0.2 pp | [-1.2, +0.6] pp | pass |
| iterative_rank6 | mmlu_pro_500 | 58.2% | 57.6% | -0.6 pp | [-2.4, +1.0] pp | inconclusive |
| complementary_rank2 | wmdp_cyber_full | 59.2% | 60.3% | +1.1 pp | [-0.1, +2.2] pp | pass |
| complementary_rank2 | cybermetric_500 | 95.6% | 94.8% | -0.8 pp | [-2.2, +0.6] pp | inconclusive |
| complementary_rank2 | mmlu_pro_500 | 58.2% | 58.2% | +0.0 pp | [-2.0, +2.0] pp | pass |

## Interpretation

- Rank 6 has the stronger refusal-removal effect, but it changes safe next-token distributions much more.
- Rank 2 retains more refusals, but its safe KL is much lower and its capability results are similar to base.
- Neither candidate caused empty harmful outputs. The uncensored judge marked zero harmful outputs as broken for both candidates.
- The evaluation did not use the final-test split and did not create a permanent checkpoint.
