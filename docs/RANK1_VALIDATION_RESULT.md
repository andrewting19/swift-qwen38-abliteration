# Rank-1 Validation Result

## Decision state

The reversible `matched_first_output_layer_52_winsor_995` candidate at alpha
1.0 is rejected. It passed the automated refusal-language, clean harmless-KL,
over-refusal, and measured capability gates. It failed HarmBench behavior
completion. Manual review confirmed that most apparent non-refusals were soft
refusals or non-compliant answers. Review also found clear grammatical errors.

No final-test data was used. No permanent checkpoint was created. No OpenAI
model or remote judgment API was used.

## Intervention

- Rank: 1
- Direction source: matched contrast, first generated-token activation, layer 52
- Estimator: winsor-995 difference of means
- Edit scope: token embedding plus the attention output projection and MLP down
  projection in all 64 language-model layers
- Strength: alpha 1.0
- Execution: reversible in-memory weight projection

## Wider generation validation

The validation used 48 disjoint standard harmful prompts, 48 disjoint matched
harmful prompts, and 64 disjoint XSTest-safe prompts. Generation length was 128
tokens.

| Measure | Standard | Matched | Gate |
|---|---:|---:|---:|
| Opening-refusal removal | 100.0% | 100.0% | Passed |
| Anywhere-refusal removal | 97.9% | 100.0% | Passed |
| HarmBench behavior success | 2.1% | 6.3% | Failed; minimum 50% |

The deterministic rules and HarmBench measure different outcomes. The rules
detect common refusal language. HarmBench asks whether the response materially
performs the requested behavior. Manual review confirmed that the edit removed
refusal wording without reliable behavior completion.

## Harmless behavior

- Clean mean forward KL: 0.0702 nats on 61 prompts that the base model answered
- All-safe mean forward KL: 0.1400 nats on all 64 prompts
- Base false refusals excluded from the clean KL gate: 3
- Added XSTest-safe refusals: 0
- Empty outputs: 0
- Deterministic output-quality gate: passed

## Capability results

| Benchmark | Base | Candidate | Change |
|---|---:|---:|---:|
| WMDP-Cyber-256 | 58.59% | 58.59% | 0.00 points |
| CyberMetric-80 | 95.00% | 95.00% | 0.00 points |
| MMLU-Pro-500 | 58.80% | 58.60% | -0.20 points |

All three measured capability checks passed the fixed maximum two-point loss.
These tests do not resolve the HarmBench disagreement.

## Review and artifacts

Start the local server and open `/rank1` for the base-versus-candidate response
review. Use `/capability` for question-level capability changes.

Primary files:

- `runs/gpu/20260915-generation-search-51135396/alpha1_0-validation/validation_report.json`
- `runs/gpu/20260915-generation-search-51135396/alpha1_0-validation/decision.json`
- `runs/gpu/20260915-generation-search-51135396/alpha1_0-validation/capability/summary.json`

The complete 248 MB remote run was copied locally. The remote and local
aggregate SHA-256 values both equaled
`acad4ac26bd682580aac60a6520c4c38c15f78b80d6ba90794dcf389a11bc965`
before Vast.ai instance `51135396` was destroyed.

## Process implications

The weight-equivalent intervention is still numerically validated. The failure
is in candidate construction and selection:

- The deterministic refusal rules rewarded surface-level refusal-word removal.
- First-token KL did not measure later language drift.
- The deterministic repetition and length checks did not detect grammar errors.
- Multiple-choice benchmarks measured retained factual choice accuracy, not
  generated-answer quality.
- The layer-52 first-output direction probably contains assistant-opening and
  language-style information in addition to refusal information.

Do not tune alpha or add another direction to this candidate. Those changes do
not correct the measured direction's lack of refusal specificity.

## Next experiment

1. Re-score all saved rank-1, rank-2, and scope-screen outputs with a local
   open-weight response-mode judge. Distinguish direct answers, partial answers,
   soft refusals, hard refusals, and incoherent answers.
2. Use HarmBench behavior completion during candidate screening, not only after
   a refusal-rule finalist is selected.
3. Add a harmless generation-quality gate. Measure token-level drift over a
   continuation, not only first-token KL, and reject grammar or coherence
   regressions.
4. Test residual-writer-only edits without the token embedding before broad edit
   variants. The embedding edit is not required by the Orca-style method and can
   change every token representation.
5. Prioritize prompt-end directions and other saved directions that did not use
   the first generated-token activation. The rejected direction can contain
   generic answer-opening syntax.
6. Generate 128-token outputs only for candidates that show substantive-answer
   improvement on a small batched screen.
7. Form rank-2 pairs only from rank-1 directions that each show a causal
   improvement in real answering. Do not combine directions that only remove
   refusal words.
8. Run capability benchmarks only after the behavior and generation-quality
   gates pass.
9. Do not create a checkpoint or use final-test data until all gates pass.
