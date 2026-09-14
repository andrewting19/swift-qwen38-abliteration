# Evaluation Plan Draft

All arms use the same prompts, sampling settings, and judge prompt. The unchanged base model is the control.

## Refusal behavior

- Use 64 held-out AdvBench prompts from `data/splits.toml`.
- Add a second safety taxonomy before the paid run.
- Use a small, fast LLM judge for hard, soft, partial, and no refusal labels.
- Manually inspect a fixed random sample and all uncertain judge cases.
- Store raw responses, judge labels, judge model ID, judge prompt, and token settings.

## Ordinary behavior

- Calculate next-token KL divergence on the 64 held-out Alpaca prompts.
- Compare representative generated answers.
- Add a general benchmark subset with fixed examples and exact scoring.

## Cyber capability

- Use legal, sandboxed benchmark tasks only.
- Prefer benchmark tasks with exact answers or executable tests.
- Keep refusal scoring separate from task correctness.
- Record correctness, completion rate, and response format failures.

## Runtime and MTP

- Measure output tokens per second with the same server settings.
- Measure MTP draft acceptance with the unchanged MTP and the edited MTP when both exist.
- Record prompt length, output length, batch size, and serving software version.

## Required comparisons

- Base against Arm A.
- Base against Arm B.
- Arm A against Arm B.
- If MTP is tested separately, edited MTP against stock MTP with the same text weights.
