# Vast.ai Runbook

This runbook does not create an instance automatically. Rental remains a manual, reviewed action.

Read `docs/PROJECT_PLAN.md` first. It defines the current authority. The current approved run stops after reversible evaluation, artifact preservation, and result review. It does not create a permanent checkpoint.

## Required offer

- One GPU with at least 75 GiB VRAM.
- BF16 support.
- CUDA 12.8 or newer.
- At least 300 GB allocated disk.
- At least 500 Mbit/s reported download speed.
- Verified host with reliability at or above 0.98.
- Maximum total price: $2.00 per hour unless reviewed again.

Preferred GPU: RTX PRO 6000 96 GB. H100 80 GB or H200 141 GB are acceptable if their total price is competitive.

The most recent read-only search on 2026-09-14 found 96 GB RTX PRO 6000 offers near $1.40 to $1.45 per hour including 300 GB storage. Offers can disappear or change. Run `infra/vast/search_offers.sh` immediately before rental.

## Disk plan

- Base BF16 cache: about 56 GB.
- Arm A checkpoint: about 56 GB.
- Arm B checkpoint: about 56 GB.
- Activations, logits, benchmark outputs, package cache, and temporary shards: reserve 50 GB.
- Safety margin: at least 80 GB.

Total allocation: 300 GB.

## Expected spend

The approved direction study and reversible evaluation are estimated at 5 to 10 hours, or about $7.00 to $14.50 at the latest observed prices. The exact hard limit must be confirmed before rental. Stop if direction validation fails. Do not add account funds automatically.

## Before rental

```bash
PYTHONPATH=src python3 scripts/pre_gpu_readiness.py
infra/vast/search_offers.sh
```

Check that no unrelated GPU task is active. Use one rental only.

## After the instance starts

1. Transfer this repository through the existing private Git transport or `scp`. Do not place access tokens in command history.
2. Run `infra/vast/remote_preflight.sh` before downloading weights.
3. Create an isolated environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[gpu,eval]'
python3 scripts/prepare_data.py
python3 scripts/prepare_matched_data.py
python3 scripts/prepare_benchmarks.py
python3 scripts/prepare_wmdp.py
```

Use an NVIDIA CUDA image with Python 3.11 or newer. The project install supplies the pinned PyTorch and Transformers versions. Run `python3 scripts/pre_gpu_readiness.py --require-gpu` again after installation to verify BF16 support and the complete runtime.

4. Record `pip freeze`, the container image identifier, `nvidia-smi`, disk space, and the Git commit in the run directory.
   Start the rental record with the exact displayed total hourly price:

```bash
python3 scripts/record_rental.py --phase start \
  --record runs/gpu/rental.json --hourly-price PRICE \
  --instance-id INSTANCE_ID --offer-id OFFER_ID
```

5. Capture activations. This is the first command that downloads and loads the checkpoint:

```bash
PYTHONPATH=src python3 scripts/capture_activations.py \
  --output-dir runs/gpu/capture \
  --acknowledge I_UNDERSTAND_THIS_LOADS_AND_EDITS_A_55GB_MODEL
```

6. Download the small capture artifacts locally before continuing.
7. Analyze all 40 direction candidates: two data sources, five layers, and four estimators.
8. Run reversible activation intervention on refusal and harmless prompts. Compare refusal change and KL. Then run the quick capability gate for the finalists.
9. Copy activations, directions, logits, responses, benchmark results, manifests, hashes, and logs to local persistent storage.
10. Verify every remote-to-local hash.
11. Destroy the Vast instance. Stopping an instance can continue storage charges.
12. Close the rental record and compare its estimate with the Vast charge:

```bash
python3 scripts/record_rental.py --phase end --record runs/gpu/rental.json
```

Do not create Arm A or Arm B in this approved run. Permanent editing requires a later user decision after the reversible result report.

## Abort conditions

- Architecture or tensor-name mismatch.
- Less than 250 GiB free disk after startup.
- GPU has less than 75 GiB VRAM.
- Non-finite activations, directions, logits, KL, or weights.
- Direction bootstrap median cosine below 0.90.
- Reversible ablation does not remove at least 90% of base refusals.
- Mean harmless KL exceeds 0.10 nats.
- Quick capability regression exceeds 2 percentage points.
- Total rental cost reaches the agreed ceiling.
