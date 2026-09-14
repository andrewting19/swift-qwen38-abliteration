# Execution Plan

The authoritative plan is `docs/PROJECT_PLAN.md`. This short file only shows the phase status.

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

## Phase 3A: Direction study and reversible evaluation — approved, not started

- Rent one suitable GPU.
- Install the pinned environment.
- Download the pinned BF16 checkpoint.
- Run live architecture checks.
- Prepare the fixed prompt files.
- Capture activations at layers 24, 32, 38, 44, and 52 in the same forward passes.
- Calculate and save all 40 direction candidates.
- Run causal activation intervention before permanent editing.
- Run validation refusal and quick capability evaluations.
- Save hashes, logs, environment data, and costs.
- Stop and destroy the rented instance.

## Phase 3B: Permanent edit — not approved

- Review the reversible results with the user.
- Produce Arm A or Arm B only after approval.
- Start every permanent arm from the unchanged base.

## Phase 4: Final artifact selection — not started

- Compare all results with the unchanged base.
- Select an arm only if refusal falls and capability remains within the agreed limits.
- Quantize only after the BF16 result passes.
