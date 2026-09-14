# Refusal-Direction Estimator Study

## What is known

Orca states that its Qwen3.8 direction is a massive-activation-masked difference of the harmful and harmless means. It does not publish the mask algorithm or threshold.

Massive activations are a small number of activation values that are much larger than normal values. Published research reports that some are nearly input-independent and act like internal bias terms. A small mismatch between the harmful and harmless samples can therefore leave a large unrelated component in the mean difference.

## Best current approximation

Winsorization is the best documented open approximation. It uses one threshold calculated from the pooled harmful and harmless activations. Values above the positive threshold are clipped to it. Values below the negative threshold are clipped to the negative threshold. The two groups must use the same threshold.

We will calculate these candidates from the same saved activations:

1. No clipping.
2. Global absolute-value quantile 0.990.
3. Global absolute-value quantile 0.995.
4. Global absolute-value quantile 0.999.

This is a candidate study. We must not call any result the Orca rule.

## Why not choose 0.995 now

The useful threshold depends on Swift's activation distribution. Before selecting a threshold, we need the real layer-38 values. We will record:

- Maximum absolute activation.
- Activation magnitude quantiles.
- Number of values changed by each threshold.
- Coordinates and token samples that contain the largest values.
- Whether the largest coordinates are stable across harmful and harmless prompts.

If the outliers occur equally in both groups, masking them may have little effect. If they occur in only a few prompts, clipping can make the direction more stable. If an outlier carries real refusal information, clipping it can make the direction worse.

## Direction-layer scan

Layer 38 is an initial heuristic, not a result. The first capture records the same last-prompt-token activation at layers 24, 32, 38, 44, and 52. All five values are collected during each forward pass, so this scan does not require five model runs.

For each layer, the analysis compares both data sources and all four estimators. A layer is not selected from cluster distance alone. The direction is measured at that one layer, then the reversible screen projects it from every target-layer output in the selected edit arm. It must have stable bootstrap directions, held-out separation, causal refusal reduction, and low harmless KL.

## Direction checks before weight editing

For every candidate direction, calculate:

- Cosine similarity with the plain direction.
- Bootstrap cosine stability under prompt resampling.
- Harmful-versus-harmless separation on held-out prompts.
- Refusal change from reversible activation ablation.
- KL divergence from reversible activation ablation on harmless prompts.

The best direction is not automatically the one with the largest cluster separation. The selected direction must reduce refusal causally while causing little normal-behavior change.

## Coordinate mask support

The code can accept an explicit Boolean coordinate mask. Masked coordinates are set to zero in the raw mean difference before normalization. We will use this only if we define and record a clear mask rule or find the Orca rule.
