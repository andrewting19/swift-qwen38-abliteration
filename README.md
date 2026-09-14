# Swift Qwen3.8 Abliteration Experiment

This private research repository prepares a controlled refusal-direction experiment for `ukisai/Swift-Qwen3.8-27b`.

No GPU has been rented. No full model weights have been downloaded. The local preflight reads only the public model configuration and tensor index.

## Current state

- The base model and dataset revisions are pinned.
- The Swift architecture contract is checked before a model can be edited.
- Two edit plans are defined.
- The refusal-direction and projection math have CPU unit tests.
- A data preparation script creates fixed, disjoint direction and evaluation splits.
- The GPU runner has an explicit acknowledgement switch. It does not run by default.
- The exact Orca massive-activation mask is not implemented because its rule and threshold are not public.

Read [docs/BUILD_REVIEW.md](docs/BUILD_REVIEW.md) before any GPU rental.

## Local checks

```bash
python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m swift_abliteration.cli preflight \
  --config configs/orca_style_full.toml \
  --output runs/preflight-orca.json
PYTHONPATH=src python3 -m swift_abliteration.cli preflight \
  --config configs/huihui_band.toml \
  --output runs/preflight-band.json
```

## Later GPU environment

Do this only after review:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[gpu]'
python3 scripts/prepare_data.py --split-file data/splits.toml
```

The future full-model command requires `--acknowledge-large-model-run`. This switch is a safety boundary. It makes an accidental local run fail before weight download.

## Scope

This project is for controlled model-behavior research. Abliteration can remove useful safeguards and can also damage normal capability. The repository keeps refusal, capability, KL-divergence, and artifact checks separate so one result cannot stand in for all of them.
