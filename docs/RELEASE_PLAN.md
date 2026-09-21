# Swift-Qwen3.8-27B-Abliterated Release Plan

Date: 2026-09-19

## Frozen candidate

- Public name: `andrewting/Swift-Qwen3.8-27B-Abliterated`
- Base: `ukisai/Swift-Qwen3.8-27b`
- Base revision: `1b30aaaf753fe5c1cb51ada2ea0367a53445359c`
- Candidate: `iterative_rank6`
- Direction bundle SHA-256:
  `83f6fc1aca0b14f7f20dfe1225cb22eff6cbd1cf40490e8d265e9b561aef5e33`
- Candidate tensor SHA-256:
  `9082e284b4bcab0612867ac5c76cbd4f434bffd4e2038be73410b38dfe75b5e8`
- Edit: alpha 1.0 on the attention output and MLP down-projection in all
  64 language layers, for 128 edited tensors.
- Token embedding: unchanged.
- MTP head: unchanged and not supported for the first release.
- Vision tower: unchanged.

Candidate selection was frozen before the final-test run. The final-test split
was used once for release evaluation and was not used to tune or replace the
candidate.

## Release stages

1. Completed: run the frozen reversible candidate on the four 64-row final-test groups and
   the 64-row XSTest-safe final partition without printing prompt text.
2. Completed: apply the same rank-6 projection to the pinned BF16 checkpoint and save all
   edited shards in float32-compute, BF16-storage form.
3. Completed: verify the edited tensor count, shard hashes, architecture, and deterministic
   output equivalence on non-final prompts.
4. Completed: add the Swift Open License v1.0, Apache 2.0 license, NOTICE, model card,
   evaluation summary, and abliteration manifest.
5. Completed: upload to the private Hugging Face staging repository, verify all files and
   hashes, then make the repository public.

## Release result

- Public URL:
  `https://huggingface.co/andrewting/Swift-Qwen3.8-27B-Abliterated`
- Hugging Face revision:
  `583d5e0442f640fc7c27911e84c0e72cc12029a9`
- Public files: 41
- Published checkpoint size: 55.6 GB
- Verified model shards: 18 of 18
- Raw final-test prompts or responses published: none
- Release GPU: destroyed after verification

## Publication rules

- Use the word `Abliterated` in the model name.
- Describe the behavior as refusal-reduced. Do not claim zero refusal or full
  uncensoring.
- Report both substantive completion and response-mode results.
- Report the safe KL drift and capability changes.
- State that MTP and vision behavior were not part of release evaluation.
- Preserve required UkisAI and Qwen license and attribution files.
- Do not upload raw evaluation prompts, raw harmful generations, or activation
  captures.
