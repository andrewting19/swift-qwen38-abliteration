# Build Review Before GPU Rental

## What exists now

The repository has four parts.

1. **A frozen experiment description.** The base model commit, dataset commits, prompt row numbers, direction layer, edit strength, and target layers are fixed in text files.
2. **A fail-closed architecture check.** The code reads the small public `config.json` and `model.safetensors.index.json`. It checks the 64-layer layout and the exact weight names. It rejects a changed or unexpected checkpoint.
3. **The mathematical core.** The code calculates a normalized difference of group means. It projects that direction out of linear output weights and embedding rows. The edit uses FP32 calculations and can store BF16 weights later.
4. **A GPU runner boundary.** Full model loading is behind an explicit command-line acknowledgement. No current command rents hardware.

## Mental model

For one direction candidate, one prompt gives one vector from one candidate layer. There are 32 harmful vectors and 32 harmless vectors. The initial GPU capture scans layers 24, 32, 38, 44, and 52 in the same forward passes.

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

- Use the selected direction layer. Layer 38 is the original heuristic candidate.
- Edit the attention or linear-attention output and MLP output in all 64 text layers.
- Edit the embedding row space.
- Edit the MTP attention and MLP output writers.
- Do not edit the vision tower.
- Total planned tensors: 131.

This matches Orca's published tensor scope. It is not an exact reproduction because Orca did not publish its direction prompt count, prompt row IDs, massive-activation mask rule, or threshold. Our plain difference of means is the baseline.

### Arm B: Huihui-style layer band

- Use the same selected direction and alpha.
- Edit attention or linear-attention output and MLP output in text layers 18 through 51.
- Do not edit embeddings, MTP, or vision.
- Total planned tensors: 68.

This isolates the effect of editing fewer residual writers.

## Decisions before GPU use

The pre-GPU choices are now frozen. The plain estimator is the baseline, and three documented winsorized estimators are candidates. Both the independent and semantic-matched contrast sets are used. Both edit arms are available, but no arm is written until a reversible screen passes. The evaluation suite, provisional thresholds, GPU minimum, disk size, and cost ceiling are fixed.

## Known limits

- A public tensor index proves names, not runtime module behavior.
- The real activation hook and weight edit still need a short full-model validation run.
- AdvBench and Alpaca are not topic-matched. Their mean difference can contain features other than refusal.
- The five candidate layers are still heuristics. The evaluation must compare stability, held-out separation, causal refusal change, and harmless KL.
- Refusal rate alone is not enough. We must also measure ordinary KL divergence, general capability, cyber capability, coherence, and speed.
- Transformers 5.17 does not load the checkpoint-only MTP module. The shard editor preserves and edits it. Draft acceptance needs a compatible server and is a later compatibility test.

## Stop point

Local preparation is complete. No GPU was reserved. No paid API was called. No model checkpoint was written.
