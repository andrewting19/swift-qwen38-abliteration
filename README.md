# Swift Qwen3.8 Abliteration Experiment

This private research repository contains a controlled refusal-direction experiment for `ukisai/Swift-Qwen3.8-27b`.

The latest GPU experiment independently reconstructed a behavior-filtered,
massive-coordinate-masked rank-1 direction from the base model. It reduced
opening refusals and kept mean harmless KL at 0.035258 nats, but it did not
produce reliable substantive answers. Most changed outputs remained soft
refusals. The candidate is rejected. No permanent checkpoint was written and
the final-test split was not used.

## Current state

- The base model, judge models, and dataset revisions are pinned.
- The Swift architecture contract is checked before a model can be edited.
- Weight-equivalent module-output and in-memory weight edits are validated.
- Independent and semantic-matched direction and holdout sets are pinned.
- Five candidate direction layers and four estimators are defined.
- Refusal, over-refusal, coherence, KL, general capability, cyber capability, and runtime checks are fixed. MTP acceptance is a later compatibility test.
- The refusal-direction and projection math have CPU unit tests.
- A data preparation script creates fixed, disjoint direction and evaluation splits.
- The GPU runners have an explicit acknowledgement switch. They do not run by default.
- The exact Orca massive-activation mask is not implemented because its rule and threshold are not public.
- The recovery code includes iterative directions, source-specific branches, alpha screening, capability checks, and aggregate-only local judge reports.
- The latest direction was reconstructed from 128 base-refused harmful cases and 128 base-answered harmless cases. It did not copy a published direction.

Read [docs/INDEPENDENT_RECONSTRUCTION_STATUS.md](docs/INDEPENDENT_RECONSTRUCTION_STATUS.md)
for the current result. Read [docs/RANK1_VALIDATION_RESULT.md](docs/RANK1_VALIDATION_RESULT.md)
for the prior rank-1 result. Read [docs/RECOVERY_POSTMORTEM.md](docs/RECOVERY_POSTMORTEM.md)
for the earlier six-direction recovery, [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)
for the full plan, and [infra/vast/RUNBOOK.md](infra/vast/RUNBOOK.md) before
another GPU rental.

## Local checks

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/pre_gpu_readiness.py
PYTHONPATH=src python3 -m swift_abliteration.cli preflight \
  --config configs/orca_style_full.toml \
  --output runs/preflight-orca.json
PYTHONPATH=src python3 -m swift_abliteration.cli preflight \
  --config configs/huihui_band.toml \
  --output runs/preflight-band.json
```

## GPU environment

Do this only after review:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 scripts/prepare_data.py --split-file data/splits.toml
python3 -m pip install -e '.[gpu,eval]'
```

Full-model commands require an explicit acknowledgement value. This switch makes an accidental local run fail before weight download.

## Scope

This project is for controlled model-behavior research. Abliteration can remove useful safeguards and can also damage normal capability. The repository keeps refusal, capability, KL-divergence, and artifact checks separate so one result cannot stand in for all of them.
