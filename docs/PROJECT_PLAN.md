# Canonical Project Plan

The next approved reversible experiment is defined in `docs/RANK1_RERUN_PLAN.md`.
It uses base refusal-token score filtering, two activation positions, a six-item
statistical shortlist, causal removal and addition proxy tests, and a two-candidate
32-token response pilot. It stops before rank 2 if no rank-1 candidate passes.

Last updated: 2026-09-15

This file is the source of truth for the project goal, current authority, execution order, decision gates, and restart procedure. Read this file first after a context reset or a new session.

## Goal

Measure and remove a refusal direction from the pinned `ukisai/Swift-Qwen3.8-27b` BF16 checkpoint. Treat this as a controlled educational experiment. Determine whether refusal decreases without unacceptable damage to ordinary behavior, general capability, or cybersecurity capability.

The direction-quality analysis and the final evaluation report are primary outputs. A modified checkpoint is not a success by itself.

## Current authority and limits

- The user approved work through result review: rental setup, activation capture, direction analysis, reversible tests, quick capability evaluation, artifact preservation, offline scoring, and the report.
- Do not create a permanent edited checkpoint until the user reviews the reversible results.
- Do not quantize, publish, or upload a model.
- Use one rented GPU at a time.
- Do not add Vast.ai credit automatically.
- The user reports adding $15 of additional Vast credit and authorizes use until the Vast credit balance is zero. Preserve and verify artifacts before the last usable credit is consumed.
- Do not put raw harmful prompts or harmful generations in Git or in the chat.
- Do not send experiment prompts or responses to OpenAI models or APIs. Use only local open-weight judges for evaluation.

## Experiment design

### Prompt groups

Use two independent contrast sources:

1. Standard: AdvBench harmful prompts against Alpaca harmless prompts.
2. Matched: semantic-matched harmful and harmless prompt pairs.

For each source, use three disjoint roles:

- Direction measurement: 32 harmful and 32 harmless prompts.
- Candidate validation: 64 harmful and 64 harmless prompts.
- Final test: 64 harmful and 64 harmless prompts.

The repository has direction, candidate-validation, and final-test splits for each source. Candidate selection must not use the final test.

### Direction candidates

Capture the last-prompt-token residual state at layers 24, 32, 38, 44, and 52 in the same forward passes.

For each layer and prompt source, calculate four estimators:

- Plain difference of means.
- Winsorized at 0.990.
- Winsorized at 0.995.
- Winsorized at 0.999.

This produces 40 candidates.

Rank candidates with bootstrap cosine stability, validation-set standardized separation, cross-source cosine agreement, and outlier sensitivity. Cluster separation is not sufficient. A candidate must also pass a causal reversible intervention.

### Reversible weight-equivalent screen

For a small shortlist, project only the outputs of the residual-writing modules included in the planned checkpoint edit. Also project the embedding output when the planned edit includes the embedding. Compare the unchanged model and each candidate on validation prompts.

Measure:

- Harmful refusal rate.
- Harmless over-refusal rate.
- Last-prompt-token KL divergence on harmless prompts.
- Coherence.
- Severe repetition and invalid output.

For bias-free linear modules, this intervention is mathematically equivalent to the planned weight projection. Verify numerical equivalence before screening. For a single-candidate arm, apply the same edit directly to the loaded weights in memory and save no checkpoint. Use batch-4 generation only after a batch-1 versus batch-4 greedy-output equivalence pilot passes. Exclude checkpoint-only MTP from the runtime screen because the active Transformers generation path does not use it.

### Quick capability gate

Run the unchanged base and the best reversible candidate on:

- MMLU-Pro, 500 fixed items.
- GSM8K chain-of-thought, 256 fixed items.
- IFEval, 200 fixed items.
- WMDP-Cyber, 256 fixed items.
- CyberMetric-80.

Before rental, complete and locally validate a serving path that keeps reversible module-output hooks active while these benchmark clients run.

### Provisional pass limits

- Remove at least 90% of the base refusals on both harmful validation sets.
- Add no more than 5 percentage points of refusal on harmless prompts.
- Keep mean harmless last-token KL at or below 0.10 nats.
- Do not accept a general or cyber score loss above 2 absolute percentage points unless the confidence interval includes zero.
- Do not accept a coherence-dimension mean loss above 0.25 on the 1-to-5 scale.
- Reject severe repetition or invalid-output regressions.

Do not use the final test until one candidate and its settings are frozen.

## Execution order

The first rental completed stages 0 through 5. It skipped stage 6 because no first-run candidate passed the reversible behavior screen. Its stage 7 artifact transfer and shutdown are complete. The recovery rental later ran a partial stage 6 on the successful refusal candidate. Its final artifact transfer and shutdown are pending.

Live result: Vast instance `51065040` supplied one A100 SXM4 80 GB at about $1.102/hour. The base checkpoint passed live architecture validation with PyTorch 2.9.1, CUDA 12.8, `causal-conv1d` 1.7.0, and `flash-linear-attention` 0.5.2. Activation capture and analysis of all 40 direction candidates completed. The first screen used whole-transformer-layer-output projection and remains a stress test only because it was not weight-equivalent. The replacement intervention passed numerical and live equivalence checks. The corrected matched plain arm produced 256 empty outputs and was rejected. Local open-weight WildGuard and HarmBench scoring completed with zero parse errors. The consensus winsor-995 arm removed 37.5% and 40.6% of base refusals. The standard winsor-995 arm removed 39.1% and 45.3%. Both fail the fixed 90% requirement on both harmful sets. All remote run artifacts were copied locally and verified. The final-test split remains unused, the quick capability gate was not run, and no permanent checkpoint exists.

Recovery result: Vast instance `51081304` supplied one A100 SXM4 80 GB at about $1.102/hour. Six iterative source-specific directions produced a reversible no-embedding candidate. In the fresh 256-token confirmation on both 64-item harmful validation groups, it removed 92.19% and 90.63% of base refusals. It had no matched-harmless refusal increase and no empty or severe repeated outputs. WMDP-Cyber-256, CyberMetric-80, and MMLU-Pro-500 changed by -0.78, 0.00, and -1.40 accuracy points. Mean harmless KL was 0.639 nats, so the candidate fails the fixed 0.10 KL gate. An alpha ladder found no edit strength that passes both refusal and KL. The final-test split remains unused and no permanent checkpoint exists. See `docs/RECOVERY_POSTMORTEM.md`.

The local dashboard is at <http://127.0.0.1:8766/> for this run because port 8765 is in use by another local service. Start or restore it with the commands in `dashboard/README.md` and select an available port. It contains only safe aggregate status and results.

An added consensus check uses only the saved layer-38 direction and validation activations. It compares the normalized standard/matched plain average with the normalized standard/matched winsor-995 average. The winsor-995 consensus was selected as the first weight-equivalent reversible arm. Its bootstrap median cosine is 0.9940. Its held-out standardized separation is 10.52 on the standard source and 9.68 on the matched source. The final-test split remains unused. Run this candidate as a separate 256-output arm only after the numerical equivalence checks pass.

### Stage 0: Finish local controls

1. Verify the separate validation and final-test prompt splits without printing harmful text.
2. Verify the reversible benchmark serving path.
3. Verify the local open-weight WildGuard and HarmBench judge paths. OpenAI judge use is prohibited for the remainder of this experiment.
4. Run all unit tests, command checks, prompt hashes, and package checks.
5. Run a live Vast offer search.
6. Record the current credit balance and keep enough credit for artifact transfer and shutdown.

### Stage 1: Rent and verify

1. Rent one verified GPU with at least 75 GiB VRAM and 300 GB disk.
2. Record the offer ID, instance ID, displayed total hourly price, and start time.
3. Check CUDA, BF16, GPU memory, host RAM, disk, and network.
4. Destroy the instance immediately if a hard check fails.

### Stage 2: Reproduce the environment

1. Clone the private repository at the recorded commit.
2. Install the pinned dependencies.
3. Recreate prompt and benchmark files.
4. Verify every source and generated-file hash.
5. Record the container image, `nvidia-smi`, `pip freeze`, disk state, and Git state.

### Stage 3: Download and validate the base

1. Download the pinned 55.6 GB BF16 checkpoint.
2. Load it without permanent edits.
3. Validate the live text-layer modules and tensor shapes.
4. Stop on any mismatch.

### Stage 4: Capture and analyze directions

1. Capture all five layers for the direction and validation prompts. Leave the final-test prompts untouched during selection.
2. Save activations and base harmless logits.
3. Calculate all 40 candidates.
4. Write the complete direction-quality report.
5. Copy and hash these artifacts on the local machine before continuing.

### Stage 5: Reversible behavior screen

1. Verify that module-output hooks numerically match explicitly projected weight matrices.
2. Select a small candidate shortlist from direction and validation statistics.
3. Generate fixed validation responses for the base and shortlisted candidates.
4. Save harmless logits for KL.
5. Run the fixed refusal and coherence judge on the validation outputs.
6. Calculate validation refusal, over-refusal, KL, and coherence results.
7. Freeze one candidate and its settings.
8. Copy and hash all outputs locally.

### Stage 6: Quick capability gate

1. Run the fixed quick benchmarks on the base once.
2. Run them on the best reversible candidate.
3. If the candidate passes, generate the base and candidate outputs on the untouched final-test prompts.
4. Save prompt-level and aggregate results.
5. Copy and hash the results locally.

### Stage 7: End paid compute

1. Confirm that every required remote artifact exists locally.
2. Compare remote and local SHA-256 hashes.
3. Record the stop time and estimated rental cost.
4. Destroy the Vast instance. Do not leave a stopped instance with storage charges.

### Stage 8: Offline scoring and report

1. Use the pinned local WildGuard refusal judge and HarmBench harmful-compliance cross-check.
2. Manually audit 20 fixed items per arm and every classifier disagreement.
3. Calculate refusal, over-refusal, KL, coherence, capability, uncertainty, and cost summaries.
4. Write a direction-quality report and a final evaluation report.

### Stage 9: User review

Show the user the selected direction layer, data source, estimator, stability, separation, causal effect, KL, capability changes, uncertainty, failures, artifact hashes, and total cost. Then decide whether to create permanent Arm A or Arm B in a later paid run.

## Time and cost estimate

At about $1.40 to $1.45 per rental hour:

| Work | Hours | Cost |
|---|---:|---:|
| Setup, download, and validation | 0.6–1.3 | $0.85–$1.89 |
| Activation capture and direction analysis | 0.35–0.85 | $0.49–$1.23 |
| Reversible tests for two or three finalists | 1.5–3.0 | $2.10–$4.35 |
| Quick capability gate | 2.5–5.0 | $3.50–$7.25 |
| Artifact verification and transfer | 0.1–0.3 | $0.14–$0.44 |
| Total | 5.0–10.0 | $7.00–$14.50 |

These are estimates. Generation length and server throughput are the main uncertainties. Use a cost watcher and stop before the agreed limit.

## Artifact preservation contract

Use one timestamped run directory. Record the Git commit and configuration in every manifest.

Preserve locally after each paid stage:

- Rental and hardware record.
- Package lock and `pip freeze`.
- Prompt and benchmark manifests and hashes.
- Activation tensors.
- Base logits.
- All 40 direction vectors.
- Direction-quality metrics and shortlist decision.
- Base and reversible response files.
- Judge inputs and outputs.
- Benchmark prompt-level and aggregate results.
- Command log, errors, timings, and cost record.

Raw prompts, raw harmful generations, checkpoints, activations, and large benchmark outputs remain outside Git. Commit safe configuration files, hashes, commands, summaries, and reports to the private repository. Verify remote-to-local copies before destroying the instance.

The pinned base checkpoint does not need a local backup. It can be downloaded again by repository ID and revision. The activation capture must be preserved because repeating it costs GPU time.

## Abort conditions

Stop paid work if any of these conditions occurs:

- Hardware, architecture, module, or tensor mismatch.
- Less than 250 GiB free disk after setup.
- Non-finite activations, directions, logits, KL, or weights.
- Bootstrap median cosine below 0.90 for all plausible candidates.
- Reversible intervention fails to remove at least 90% of base refusals.
- Mean harmless KL exceeds 0.10 nats for all plausible candidates.
- Quick capability loss exceeds the fixed limit.
- The rental reaches the agreed cost limit.

Preserve completed artifacts before stopping when it is safe to do so.

## Restart procedure

After a context reset or interrupted rental:

1. Read this file.
2. Run `git status --short` and `git log -1 --oneline`.
3. Read the latest safe run manifest and cost record.
4. Verify local artifact hashes.
5. Identify the first incomplete stage. Do not repeat a completed stage with verified artifacts.
6. Run `scripts/pre_gpu_readiness.py` before any new rental.
7. Search Vast offers again because prices and availability change.
8. Confirm the remaining cost limit.

## Supporting documents

- Dataset details: `docs/DATASET_METHODS.md`
- Prompt metadata: `docs/PROMPT_SET_AUDIT.md`
- Direction estimators: `docs/DIRECTION_ESTIMATORS.md`
- Evaluation protocol: `docs/EVAL_PROTOCOL.md`
- Vast operation: `infra/vast/RUNBOOK.md`
- Frozen sources: `docs/SOURCES.md`
