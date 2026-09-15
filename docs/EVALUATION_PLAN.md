# Evaluation Plan

All arms use the same prompts, sampling settings, and judge prompt. The unchanged base model is the control.

## Refusal behavior

- Use 64 held-out AdvBench prompts from `data/splits.toml`.
- Use the separate semantic-matched harmful holdout.
- Use two GPT-5.6 Luna high-reasoning passes for hard, soft, partial, and full-answer labels. Use Luna xhigh only for binary refusal disagreements.
- Manually inspect a fixed random sample and all uncertain judge cases.
- Store raw responses, judge labels, judge model ID, judge prompt, and token settings.

## Ordinary behavior

- Calculate next-token KL divergence on the 64 held-out Alpaca prompts.
- Compare representative generated answers.
- Use the fixed general benchmark subsets in `configs/evaluation.toml`.

## Cyber capability

- Use legal, sandboxed benchmark tasks only.
- Prefer benchmark tasks with exact answers or executable tests.
- Keep refusal scoring separate from task correctness.
- Record correctness, completion rate, and response format failures.

## Runtime and MTP

- Measure output tokens per second with the same server settings.
- Measure MTP draft acceptance only if a serving engine supports this checkpoint's MTP module. Transformers 5.17 ignores that module.
- Record prompt length, output length, batch size, and serving software version.

## Required comparisons

- Base against Arm A.
- Base against Arm B.
- Arm A against Arm B.
- If a compatible MTP server is available, compare edited MTP against stock MTP with the same text weights.
