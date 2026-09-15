# Reversible Screen Status

Date: 2026-09-15

## Whole-residual stress test

The first implementation projected the refusal direction from each complete transformer-layer output. This is stronger than the planned checkpoint weight edit and is not a valid weight-equivalent selection test.

Preserve these results only as a stress test:

- Candidate: `matched_layer_38_plain`
- Harmful refusal rate: 100.0% on both validation sources
- Harmless over-refusal rate: 87.5% standard and 93.8% matched
- Empty outputs: 122 of 256
- Standard harmless mean last-token KL: 7.432 nats
- Standard harmless mean coherence: 2.086 of 5

The `standard_layer_38_plain` stress arm was stopped after 47 of its first 64 records. Its partial file SHA-256 is `e05eab13dde0ea7341cb074f10e5a9ad021c221c7b6d4b96dbc7995c76fd96ae`. Do not score it as a completed arm.

## Weight-equivalent intervention validation

The replacement intervention hooks only the token embedding output and the attention and MLP output projections included in the checkpoint edit. It projects only each module's output, not the complete residual stream.

The local numerical unit test compares the hooks with an explicitly projected copy of the same weight matrix. It also tests biased linear modules and preserves the bias term.

The live check passed on the target model:

- Runtime writer modules: 129
- Biased runtime modules: 0
- Input and output embeddings tied: no
- MTP available to this Transformers model: no
- MTP calls during a one-token generation: 0
- Sampled modules: embedding, linear-attention output, MLP output, and full-attention output
- Minimum sampled cosine similarity: 0.9999949
- Maximum sampled absolute BF16 difference: 0.03125
- Live check artifact SHA-256: `269eed675cf0afd33edbfcaf2eda08ee222639cb50c7ced4181167a7be6377db`

The selected `consensus_layer_38_winsor_995` weight-equivalent screen is active. The final-test split remains unused. No permanent checkpoint exists.
