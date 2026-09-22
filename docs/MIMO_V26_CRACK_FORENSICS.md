# MiMo-V2.6 CRACK weight forensics

Date: 2026-09-22

Status: preliminary public-checkpoint analysis. This document records measured
properties of a third-party model. It does not claim access to dealignai's
private probe data or CRACK implementation.

## Question

The public model card for
[`dealignai/MiMo-V2.6-Flash-RL-UNCENSORED`](https://huggingface.co/dealignai/MiMo-V2.6-Flash-RL-UNCENSORED)
says that refusal was removed at the weight level. It also says that the
repository does not include the refusal vectors, target-layer indices,
strength schedule, probe artifacts, or surgery records.

The edited weights are public. This permits a direct comparison with the
declared base checkpoint. A weight comparison can show the final edit. It
cannot uniquely recover the prompt set or the process that selected the edit.

## Checkpoints

| Role | Repository | Revision used |
| --- | --- | --- |
| Base | `XiaomiMiMo/MiMo-V2.6-Flash-RL` | `5711b268169967567844e1e560e8a3966da959b1` |
| Edited | `dealignai/MiMo-V2.6-Flash-RL-UNCENSORED` | `0bdb11d7f37d3f6939e53f822c2bff518e4852fa` |

These were the public revisions available during this analysis. Later model
updates can produce different results.

## File-level findings

The model uses 64 expert-parallel main weight files. The comparison found:

- `model_pp0_ep0_shard0.safetensors` differs.
- The other 63 `model_pp0_ep*_shard0.safetensors` files have the same public
  LFS SHA-256 values as the base.
- `model_mtp.safetensors` is byte-identical to the base.
- `dflash/dflash_draft_model.safetensors` is byte-identical to the base.
- `model.safetensors.index.json` and the DFlash index are identical.

This agrees with the model card's statement that the DFlash drafter and MTP
weights were preserved. It also limits the language-model edit to the shard
that contains shared decoder tensors and the first expert-parallel partition.

The changed shard has 1,969 tensors in both checkpoints. Tensor names, shapes,
and data types match. It contains 48 language-model attention output matrices:

```text
model.layers.<0..47>.self_attn.o_proj.weight
dtype: BF16
shape: [4096, 8192]
```

## Layer-band finding

A 64 KiB middle-window comparison of every language attention output matrix
gave this pattern:

- Layers 0 through 33: sampled windows are identical.
- Layers 34 through 47: sampled windows differ.

The model card states that only decoder attention output projections were
changed. The measured layer pattern therefore supports a late 14-layer,
attention-output-only edit. The window scan is not a full checksum of the
unchanged tensors. A complete tensor-level audit is still required before
using the word "only" as an independently verified statement.

## Rank analysis

For each changed attention output matrix, the full BF16 tensor was fetched by
HTTP range request. Let:

\[
D_i = W_{i,\mathrm{base}} - W_{i,\mathrm{edited}}.
\]

Power iteration measured the dominant singular component of each `D_i`.

| Layer | `||D||F / ||W||F` | Energy in top rank-1 component |
| ---: | ---: | ---: |
| 34 | 6.020% | 99.891% |
| 35 | 5.658% | 99.848% |
| 36 | 8.169% | 99.364% |
| 37 | 4.283% | 99.794% |
| 38 | 6.769% | 99.881% |
| 39 | 9.522% | 96.891% |
| 40 | 6.464% | 99.732% |
| 41 | 8.190% | 99.830% |
| 42 | 11.170% | 96.918% |
| 43 | 10.075% | 97.945% |
| 44 | 10.305% | 99.471% |
| 45 | 11.481% | 96.569% |
| 46 | 11.378% | 98.994% |
| 47 | 14.461% | 96.320% |

The mean top-rank-1 energy fraction is 98.675%. The final changes are therefore
well described as one dominant rank-one update per edited layer, plus smaller
components. BF16 rounding can contribute to the smaller components.

## The directions are layer-specific

The dominant left singular vector is the output-space direction of each
rank-one matrix difference. If one direction had been applied to every layer,
these vectors would have absolute cosine similarity near 1.

The measured absolute cosine similarities between adjacent layers range from
0.030 to 0.518. Their mean is 0.229 and their median is 0.211. The layer-34 and
layer-47 vectors have absolute cosine similarity 0.531.

This is strong evidence that the final checkpoint does not use one shared
refusal vector in all 14 layers. It uses different rank-one directions at
different layers, or an equivalent construction that produces that result.

## It is not a uniform canonical projection

Canonical left-side abliteration of an attention output matrix has the form:

\[
W'_i = (I - \alpha_i r_i r_i^T)W_i.
\]

Its difference is:

\[
D_i = \alpha_i r_i(r_i^T W_i).
\]

Under this formula, the dominant right singular vector of `D_i` must align
with `W_i^T r_i`. Four inspected layers gave left-projection alignment values
from 0.555 to 0.893, except that some other layers were close to 1. This is not
consistent with one exact, uniform implementation of the canonical formula.

The most conservative interpretation is:

- each selected layer received a dominant rank-one outer-product update;
- the output and input-side vectors can vary independently by layer;
- some layers can approximate ordinary orthogonal projection;
- other layers use a more general rank-one edit, or contain additional effects
  from the private pipeline and BF16 serialization.

A useful approximation of the final operation is therefore:

\[
W'_i \approx W_i - \sigma_i u_i v_i^T,
\qquad i \in \{34,\ldots,47\}.
\]

This describes the observed weight delta. It does not prove how `u_i`, `v_i`,
or `sigma_i` were selected.

## Likely recipe

The public CRACK description and the measured deltas support this educated
reconstruction:

1. Run harmful and harmless probes in the base model.
2. Measure refusal-related signals separately at many layers or pathways.
3. Test candidate interventions against refusal and capability evaluations.
4. Select the late layer band 34 through 47.
5. Apply one main rank-one update to each selected attention output matrix.
6. Use layer-specific directions and effective strengths.
7. Leave MoE expert weights, embeddings, MTP, DFlash, vision, and audio paths
   unchanged, subject to a future full tensor audit.
8. Evaluate thinking-on and thinking-off modes separately.

This differs from the released Swift-Qwen3.8 experiment in this repository.
That release projects a shared rank-6 subspace out of both attention output and
MLP down-projection weights in all language layers.

## What the weights cannot establish

The public checkpoint cannot uniquely reveal:

- the harmful and harmless prompt sets;
- whether examples were filtered by actual model behavior;
- the activation token positions;
- the direction estimator;
- the layer-scoring objective;
- the search procedure for the layer band and strengths;
- whether the smaller non-rank-one components are deliberate or serialization
  effects.

The final deltas can be recovered and replayed. That would reproduce this
specific edited checkpoint. It would not provide a method for finding the same
kind of edit in a new model.

## Reproduction tool

The public script
[`scripts/inspect_remote_safetensor_edit.py`](../scripts/inspect_remote_safetensor_edit.py)
uses HTTP range requests. It does not download full checkpoint shards.

Install the small optional dependency:

```bash
python -m pip install -e '.[forensics]'
```

Run the low-bandwidth layer scan:

```bash
python scripts/inspect_remote_safetensor_edit.py \
  --mode scan \
  --layers 0-47
```

Run the full rank-one analysis for the edited band:

```bash
python scripts/inspect_remote_safetensor_edit.py \
  --mode rank1 \
  --layers 34-47
```

The rank-one analysis transfers about 1.9 GB because it reads both versions of
14 matrices. It keeps the model shards remote.

## Sources

- [Tweet announcing the model](https://x.com/dealignai/status/2102398464558784772)
- [Edited model card](https://huggingface.co/dealignai/MiMo-V2.6-Flash-RL-UNCENSORED)
- [Declared base checkpoint](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)
- [Public CRACK description](https://dealign.ai/crack.html)
