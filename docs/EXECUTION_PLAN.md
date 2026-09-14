# Execution Plan

## Phase 1: Local preparation — complete

- Freeze model and dataset revisions.
- Freeze prompt indices.
- Validate public model metadata and weight names.
- Test direction and projection math on small arrays.
- Define two controlled edit arms.

## Phase 2: Review — complete

- Review the files and assumptions.
- Decide the first GPU experiment.
- Define evaluation thresholds.
- Add a semantic-matched contrast set.
- Add a five-layer direction scan and reversible activation intervention.
- Validate the pinned runtime, processor, and model class without model weights.

## Phase 3: GPU validation — not started

- Rent one suitable GPU.
- Install the pinned environment.
- Download the pinned BF16 checkpoint.
- Run live architecture checks.
- Prepare the fixed prompt files.
- Capture activations at layers 24, 32, 38, 44, and 52 in the same forward passes.
- Calculate and save the refusal direction.
- Run causal activation intervention before permanent editing.
- Produce Arm A and, if approved, Arm B.
- Run refusal and capability evaluations.
- Save hashes, logs, environment data, and costs.
- Stop and destroy the rented instance.

## Phase 4: Artifact selection — not started

- Compare all results with the unchanged base.
- Select an arm only if refusal falls and capability remains within the agreed limits.
- Quantize only after the BF16 result passes.
