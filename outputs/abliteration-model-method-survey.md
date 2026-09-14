# Abliterated model method survey

Research date: 2026-09-04

## Main finding

The public model list is much larger than the earlier survey. However, many model repositories are quantizations, mirrors, merges, or later refinements of a smaller number of source checkpoints. A repository with many downloads is not always a separate abliteration method.

For Qwen3.8-27B, the main method families are:

1. OrcaRouter: one masked difference-of-means direction and a direct full-weight edit.
2. Heretic: automated search over layer and strength parameters, with refusal count and KL divergence as the joint objective.
3. Heretic ARA and iterative PRE: higher-rank or repeated low-rank versions of Heretic.
4. Pliny/OBLITERATUS: multi-direction SVD, LEACE, weight blending, and targeted iterative passes.
5. Huihui: a simple difference-of-means implementation with a selected layer window.
6. HauhauCS: a popular aggressive checkpoint with no official complete method record.
7. Blackfrost: a proprietary direction-bank and scaling process with limited public detail.
8. Manual single-direction builds, such as Philbert and hotdogs, with fixed layer and strength choices.

The original technical basis is Arditi et al. The paper found that refusal can often be controlled by a one-dimensional residual-stream direction. Removing that direction from residual activations or residual-writing weights reduces refusal. The result was shown on 13 open chat models up to 72B. See the [Arditi et al. paper](https://arxiv.org/abs/2406.11717) and [Maxime Labonne's implementation article](https://huggingface.co/blog/mlabonne/abliteration).

Important term: most of these methods are not post-training. They are training-free weight edits. Some authors later merge a LoRA, use DPO or SFT, or quantize the edited checkpoint. Those later steps are post-training or packaging steps.

## Popularity snapshot

The counts below are Hugging Face downloads in the last month on 2026-09-04. Hugging Face counts file requests. A split GGUF repository with many quant files can get a high count. The count is not a count of users. It is useful only as one popularity signal.

| Repository | Downloads | What it really is | Method source |
|---|---:|---|---|
| [HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive) | 2,543,857 | Source checkpoint and quant set | HauhauCS Aggressive; full method not public |
| [JonathanColetti/Qwen3.8-27B-Uncensored-GGUF](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF) | 2,395,758 | Quant package | Jonathan Coletti Heretic source |
| [huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF) | 2,069,464 | Quant package | Huihui direct layer-window edit |
| [HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF) | 1,463,966 | Quant package | HauhauCS Aggressive; full method not public |
| [0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF](https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF) | 1,399,307 | Mixed repository: old Heretic file and newer RVN files | trohrbaugh ARA plus two more ARA passes |
| [OBLITERATUS/Qwen3.8-27B-OBLITERATED](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED) | 928,393 | BF16 source plus many derivatives | Pliny/OBLITERATUS V3 |
| [mlabonne/Qwen3-30B-A3B-abliterated](https://huggingface.co/mlabonne/Qwen3-30B-A3B-abliterated) | 404,481 | Earlier Qwen source checkpoint | Labonne directional abliteration family |
| [orcarouter/Qwen3.8-27B-Uncensored-FP8](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8) | 343,382 | FP8 source release | OrcaRouter single-direction edit |
| [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) | 324,670 | Quant package | Blackfrost private process |
| [orcarouter/Qwen3.8-27B-Uncensored-GGUF](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF) | 276,706 | Quant package | Same OrcaRouter source |
| [orcarouter/Qwen3.8-27B-Uncensored-MLX](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX) | 134,532 | MLX quant package | Same OrcaRouter source |
| [outsourc-e/Qwen3.8-27B-Unleashed-GGUF](https://huggingface.co/outsourc-e/Qwen3.8-27B-Unleashed-GGUF) | 79,585 | Dynamic GGUF package | Jonathan Coletti Heretic source |

Other popular repositories, such as DavidAU releases, often combine an abliterated source with model merging, NEO tuning, task fine-tuning, or a new quant recipe. They can be good inference packages. They are poor sources for learning one abliteration method because more than one change is present.

## Method details

### 1. OrcaRouter Qwen3.8

Public method record: high.

The [OrcaRouter method article](https://www.orcarouter.ai/blog/how-abliteration-works) gives a clear Qwen3.8 recipe:

- Use 32 harmful and 32 harmless prompts.
- Capture the last-token residuals.
- Use a massive-activation-masked mean difference.
- Use one normalized direction, `k=1`.
- Estimate the direction at layer 38, which is `round(0.6 * 64)`.
- Apply the edit in FP32.
- Edit 131 residual writers: 17 `self_attn.o_proj`, 48 `linear_attn.out_proj`, 65 `mlp.down_proj`, and the embedding row space.
- Leave the vision tower unchanged.
- Edit the MTP output projections with the language model instead of copying the stock MTP head back.
- Quantize the result to the official block-FP8 form.

OrcaRouter reports very large refusal reductions with small benchmark changes. These are release-author results. The article also reports that 99.9% of the final FP8 codes match the official quantized base. This check is useful because it tests whether the quantization step changed more than intended.

This is the best simple control for a new Qwopus experiment. It is direct, easy to explain, and has few free parameters.

The earlier Abliteration Station work linked the [OrcaRouter X release post](https://x.com/OrcaRouter/status/2089385980080148726). That post is for the MLX package. The method article is the better source for the weight edit.

### 2. Jonathan Coletti Heretic and Unleashed

Public method record: high.

The [Jonathan Coletti source card](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored) says:

- Run Heretic in BF16 for 200 optimization trials.
- Co-minimize refusal count and KL divergence from the base.
- Edit only the 64 attention or linear-attention output projections and the 64 MLP down projections.
- Merge the selected LoRA into BF16 base weights.
- Copy all 15 MTP tensors from the base after the merge.
- Leave vision unchanged.

The published point has 12 refusals in a 100-prompt held-out set and first-token KL of 0.1191. Its four-task capability mean is 0.5 percentage point below the base. The card also publishes the full refusal/KL Pareto front. This is one of the best public records because it shows the rejected trade-offs, not only the selected result.

The popular [Unleashed GGUF](https://huggingface.co/outsourc-e/Qwen3.8-27B-Unleashed-GGUF) is not a separate abliteration method. It uses Jonathan Coletti's Heretic weights. It adds an Unsloth Dynamic 3.0 quant recipe, MTP retention, a vision projector, and detailed inference tests. Its [X release post](https://x.com/outsource_/status/2090609593478959282) reports 0/20 refusals and 82.98% MMLU for the exact Q3 quant. The card correctly says that the MMLU difference includes both weight-edit loss and quantization loss.

### 3. Heretic ARA and repeated PRE

Public method record: high.

[Heretic](https://github.com/p-e-w/heretic) automates directional abliteration with Optuna TPE. It searches layer ranges and edit strengths. Its objective reduces refusal-keyword hits while it limits KL divergence from the parent. Current Heretic can use different row-normalization and rank methods.

Two Qwen3.8 examples are useful:

- [trohrbaugh/Qwen3.8-27B-heretic-ara](https://huggingface.co/trohrbaugh/Qwen3.8-27B-heretic-ara) uses an arbitrary-rank ablation method. It edits layers 26 through 56. It reports 0/100 refusals and KL 0.0535. It gives the exact preservation, steering, overcorrection, and neighbor parameters.
- [gjtgjt/Qwen3.8-27B-heretic-r1n](https://huggingface.co/gjtgjt/Qwen3.8-27B-heretic-r1n) uses repeated true rank-1 PRE edits. It recomputes the direction after each accepted round. Round 1 reduces refusal hits from 98 to 25 at cumulative KL 0.0525. Round 2 reaches 18 at KL 0.0931. Round 3 reaches 8, but it is rejected because KL is 0.1983 and exceeds the fixed 0.1 cap.

The 0bserverx RVN release starts from the trohrbaugh ARA checkpoint and applies two more ARA passes. Its card reports 0-1 refusals and KL 0.0085. The repository also keeps an old file for download-count continuity. Thus, its 1.4 million download count combines more than one artifact generation.

For Qwopus, iterative PRE is a good second experiment after the Orca-style control. It has a clear stop rule: do not accept the next round if cumulative KL is above the preset cap.

### 4. Pliny and OBLITERATUS

Public method record: medium.

Pliny's [Qwen3.8 OBLITERATED card](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED) describes three versions:

- V1: five iterative SVD directions with low regularization. The card reports 0% refusal and a 6-point MMLU loss.
- V2: build one three-direction SVD edit at regularization 0.08 and one three-direction LEACE edit at regularization 0.06. Blend 60% LEACE with 40% SVD. Restore MTP and vision from the base. The card reports a 0.28-point MMLU loss but some soft deflections.
- V3: start from V2. Add a gentle two-direction SVD refinement at regularization 0.04. Make a targeted three-direction SVD edit at regularization 0.01 with a focused corpus. Blend the two results 50/50. Restore MTP and vision. The card reports no hard refusals or soft deflections and a 2.12-point MMLU loss.

The release card has much more method detail than most model cards. However, the project's own [research summary](https://github.com/elder-plinius/OBLITERATUS/blob/main/docs/executive_research_summary.md) says that the V2 contributor results were not independently reproduced. It says that raw benchmark outputs, prompt-level evidence, exact model revisions, environment details, and artifact hashes were not included. It also says that the 60/40 search used a small MMLU sample and did not include a same-method blend control. The [Qwen3.8 research roadmap](https://github.com/elder-plinius/OBLITERATUS/blob/main/docs/QWEN38_27B_RESEARCH_ROADMAP.md) documents one promotion-grade control that still had 92% refusal. This does not disprove the release checkpoint. It means that the public reproduction record is incomplete.

For Qwopus, do not start with the Pliny blend. First run each SVD and LEACE arm alone. Then add a same-method blend control. Only then test the SVD/LEACE blend. This design can show if any gain comes from complementary methods or only from weight averaging.

### 5. Huihui

Public method record: medium-low.

The [Huihui Qwen3.8 card](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) points to [Sumandora's implementation](https://github.com/Sumandora/remove-refusals-with-transformers). That implementation:

- Randomly samples 32 harmful and 32 harmless prompts.
- Captures one-token hidden states at the last prompt position.
- Uses layer index `int(number_of_layers * 0.6)` to calculate one difference-of-means direction.
- Normalizes that direction.
- Uses inference hooks or a weight edit to remove the direction.

For the current Qwen3.8 checkpoint, Huihui says only layers 18 through 51 are ablated. MTP and vision are not modified. The card gives no refusal benchmark, KL value, capability A/B, random seed, or complete Qwen3.8 build command. It is popular, but it is not the best reproduction source.

### 6. HauhauCS Aggressive

Public method record: low.

The [HauhauCS Qwen3.8 card](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF) reports 0 refusals in 465 tests and says that text, MTP, and vision support are retained. It does not publish the direction corpus, direction rank, edited tensor list, layer schedule, strength values, code, base revision, raw refusal outputs, or capability A/B.

Community reconstructions describe this family as a multi-direction Reaper or Heretic-like process with layer-dependent curves and more tensor types. That is not an official method record. The [0xKitkat Qwen3.8 reconstruction](https://huggingface.co/0xKitkat/Qwen3.8-27B-Uncensored-Aggressive) uses a rank-4 token-derived basis, a late-layer tent curve, residual writers, some residual readers, and row-norm preservation. Treat this as a reconstruction, not proof of the HauhauCS process.

HauhauCS is important for popularity and output comparison. It is not suitable as the primary reproducible method for Qwopus.

### 7. Blackfrost

Public method record: low.

The [Blackfrost BF16 card](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16) says it uses a weight-level refusal-surface direction modification. It states that the model is not an SFT, DPO, LoRA, merge, pruning, or quantization result. It reports 11 remaining refusals in a 450-case sequential manual funnel. The exact direction bank, corpus, layer schedule, strength schedule, and build code are not public. The default template also has an embedded operational system prompt, so weight effects and prompt effects must be tested separately.

### 8. hotdogs and Philbert manual controls

Public method record: high for the weight edit; small evaluation sets.

The [hotdogs Qwen3.8 card](https://huggingface.co/hotdogs/Qwen3.8-27B-abliterated) is a clean single-direction reproduction:

- 32 harmful and 32 harmless prompts.
- Choose hidden-state position 46.
- Sweep edit strength over 0 to 2.5.
- Select strength 1.2.
- Remove the rank-1 component from every detected residual writer.
- Verify both structural removal and behavior.

The source code is in [LLM-abliterate](https://github.com/nanofatdog/LLM-abliterate). This model is not among the top downloads, but it has one of the best direct reproduction records.

The [Philbert Qwen3.8 card](https://huggingface.co/philbert440/Qwen3.8-27B-Uncensored-Aggressive) uses one direction at layer 28 with alpha 1.15. It reports that alpha 1.24 over-edited the model and increased confabulation and reasoning loss. This is useful evidence that maximum refusal removal is not always the best model.

## What the popular older models added

The Qwen3.8 models use methods that came from older model families:

- FailSpy's Llama 3 releases used orthogonalized BF16 weights and refined the original single-direction process. See [Llama-3-70B-Instruct-abliterated-v3](https://huggingface.co/failspy/Llama-3-70B-Instruct-abliterated-v3).
- Maxime Labonne made the method common with a full notebook and model releases. The basic method tests candidate residual directions and then edits embeddings, attention outputs, and MLP outputs. Later Gemma releases used a direction for each layer and a weighted layer profile.
- GrimJim added projected and norm-preserving forms. These try to remove the refusal-specific component while preserving helpful directions and row norms. See [Projected Abliteration](https://huggingface.co/blog/grimjim/projected-abliteration) and [Norm-Preserving Biprojected Abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration).
- Heretic converted manual tuning into a Pareto search with refusal and KL objectives.
- OBLITERATUS added multi-direction extraction, LEACE, iterative passes, and model blending.

## Recommended Qwopus experiment

Use the full BF16 or FP16 safetensors as the source. Do not edit a GGUF. Quantize only after you select a checkpoint.

Use four controlled arms:

| Arm | Edit | Purpose |
|---|---|---|
| A | Orca-style one direction at about layer 38; edit all Qwen residual writers | Simple baseline |
| B | Heretic PRE with a fixed cumulative KL cap of 0.10 | Automated capability-preserving search |
| C | Heretic ARA or full row normalization | Test a higher-rank edit |
| D | OBLITERATUS SVD and LEACE arms, then a blend with same-method controls | Test the Pliny hypothesis |

Keep these factors separate:

- Direction data: fixed train, tuning, and held-out prompt identities.
- Thinking mode: evaluate thinking on and off.
- MTP: test one checkpoint with stock MTP restored and one with MTP edited consistently. Do not mix the two results.
- Vision: hash the vision tensors before and after the edit.
- Quantization: evaluate the BF16 checkpoint first. Then evaluate the exact final quant.
- Template: use the same chat template and system prompt for base and edited models.
- Randomness: record seed, model revision, tokenizer revision, package lock, GPU, and dtype.

Minimum acceptance record:

1. Refusal counts on a held-out set, with hard refusals and soft deflections scored separately.
2. KL against the exact base, with the KL definition stated.
3. Base versus edited capability results in the same run.
4. Code, math, instruction-following, tool-use, long-context, and vision tests.
5. Degeneration tests for loops, repetition, confabulation, and language mixing.
6. Tensor diff inventory that lists every changed tensor.
7. Hashes for base, edited BF16, MTP, vision, and final quant artifacts.

The best first experiment is Arm A versus Arm B. These two runs give a clear answer about whether Qwopus needs an automated Pareto search. Add Pliny-style blending only if the single-direction and Heretic results leave an important refusal or capability problem.

## Evidence grades

| Family | Method detail | Reproduction data | Evaluation quality |
|---|---|---|---|
| OrcaRouter | High | Medium-high | Medium; mostly author-reported |
| Jonathan Coletti Heretic | High | High | Medium-high |
| trohrbaugh/gjtgjt Heretic variants | High | High | Medium |
| Pliny/OBLITERATUS | High recipe detail | Medium-low for exact released result | Mixed; strong claims and explicit caveats |
| hotdogs | High | High | Medium; new and less used |
| Huihui | Medium-low | Medium-low | Low |
| HauhauCS | Low | Low | Low-medium; large claimed refusal set, few raw details |
| Blackfrost | Low | Low | Medium-low |
| Unleashed | High for quant and runtime; inherited for abliteration | High for packaging | Medium for exact quant |

## Direct X links recovered from the earlier work

- [OrcaRouter MLX release](https://x.com/OrcaRouter/status/2089385980080148726)
- [Unleashed first release post](https://x.com/outsource_/status/2090609593478959282)
- [Unleashed later results and download post](https://x.com/outsource_/status/2093706271350231313)
- [BlackwellBoy Qwen3.8 runtime article post](https://x.com/Blackwellboy/status/2090089804121907333)

The X posts are useful for finding releases. The model cards, repository code, raw evaluation files, and checkpoint tensor differences are stronger sources for the abliteration method.
