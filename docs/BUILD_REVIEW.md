# Build Review Before GPU Rental

## What exists now

The repository has four parts.

1. **A frozen experiment description.** The base model commit, dataset commits, prompt row numbers, direction layer, edit strength, and target layers are fixed in text files.
2. **A fail-closed architecture check.** The code reads the small public `config.json` and `model.safetensors.index.json`. It checks the 64-layer layout and the exact weight names. It rejects a changed or unexpected checkpoint.
3. **The mathematical core.** The code calculates a normalized difference of group means. It projects that direction out of linear output weights and embedding rows. The edit uses FP32 calculations and can store BF16 weights later.
4. **A GPU runner boundary.** Full model loading is behind an explicit command-line acknowledgement. No current command rents hardware.

## Mental model

For the direction calculation, one prompt gives one vector from layer 38. There are 32 harmful vectors and 32 harmless vectors.

```text
mean(harmful vectors) - mean(harmless vectors) = raw direction
raw direction / its length = unit refusal direction r
```

For one residual-writing matrix `W`, the edit removes the part of each output that points along `r`:

```text
W_new = W - alpha * r * (r_transpose * W)
```

For the token embedding matrix `E`, the hidden dimension is on the other matrix axis:

```text
E_new = E - alpha * (E * r) * r_transpose
```

`alpha = 1` removes the full measured component. This does not prove that all refusal behavior is gone. It only removes the measured direction from the selected writers.

## The two planned arms

### Arm A: Orca-style full topology

- Calculate one direction at layer 38.
- Edit the attention or linear-attention output and MLP output in all 64 text layers.
- Edit the embedding row space.
- Edit the MTP attention and MLP output writers.
- Do not edit the vision tower.
- Total planned tensors: 131.

This matches Orca's published tensor scope. It is not an exact reproduction because Orca did not publish its massive-activation mask rule or threshold. Our current direction is a plain difference of means.

### Arm B: Huihui-style layer band

- Use the same direction and alpha.
- Edit attention or linear-attention output and MLP output in text layers 18 through 51.
- Do not edit embeddings, MTP, or vision.
- Total planned tensors: 68.

This isolates the effect of editing fewer residual writers.

## Decisions to make before GPU use

1. Keep the plain mean difference as the reproducible baseline, or design and name our own activation mask.
2. Keep the independent AdvBench and Alpaca groups, or build a topic-matched contrast set.
3. Run both edit arms in one GPU rental, or run Arm A first.
4. Set the exact evaluation suite and pass thresholds.
5. Select the provider and GPU only after the measured disk and memory plan is complete.

## Known limits

- A public tensor index proves names, not runtime module behavior.
- The real activation hook and weight edit still need a short full-model validation run.
- AdvBench and Alpaca are not topic-matched. Their mean difference can contain features other than refusal.
- A fixed layer-38 direction is a heuristic. The evaluation must compare layer separation and causal refusal change.
- Refusal rate alone is not enough. We must also measure ordinary KL divergence, general capability, cyber capability, coherence, speed, and MTP draft acceptance.

## Stop point

The work stops here until review. No GPU was reserved. No paid API was called. No model checkpoint was written.
