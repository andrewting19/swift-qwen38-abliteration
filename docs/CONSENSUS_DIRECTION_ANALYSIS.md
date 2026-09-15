# Layer-38 Consensus Direction Analysis

Date: 2026-09-15

This analysis used only the saved direction-measurement and candidate-validation activations. It did not use the final-test split.

## Method

Two candidates were calculated:

1. Unit-normalized average of `standard_layer_38_plain` and `matched_layer_38_plain`.
2. Unit-normalized average of `standard_layer_38_winsor_995` and `matched_layer_38_winsor_995`.

For each of 500 bootstrap samples, each source was resampled independently. Its direction was recalculated. The two source directions were then normalized and averaged. Stability is the cosine similarity between each bootstrap consensus and the full-data consensus.

## Results

| Candidate | Bootstrap median cosine | Standard held-out separation | Matched held-out separation | Source-direction cosine |
|---|---:|---:|---:|---:|
| Plain consensus | 0.994094 | 8.756619 | 9.026501 | 0.934956 |
| Winsor-995 consensus | 0.994036 | 10.520647 | 9.676396 | 0.938303 |

The cosine similarity between the two consensus candidates is 0.974342.

## Selection

Select only `consensus_layer_38_winsor_995` for one added reversible arm with 256 maximum new tokens per output. Its bootstrap stability is effectively equal to the plain consensus. Its held-out separation is higher on both sources. Its two source directions also agree slightly more closely.

Do not create a permanent checkpoint from this selection. The reversible behavior and capability results are still required.

Safe artifact hashes:

- Consensus direction file: `38baf0ca230cb87ae0176636359216280896e3d7aa12d2b545f740b9c430ae93`
- Consensus report: `a5323368f216ba9bd4d4217c79c83f491120f2dd6a76b3b1b3e88417e7c64022`
