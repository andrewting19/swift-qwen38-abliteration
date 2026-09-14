# Prompt Set Audit Without Prompt Text

Raw prompt text is not tracked by Git. This page contains only source names, roles, counts, and SHA-256 hashes.

| Set | Source | Role | Count | SHA-256 |
|---|---|---|---:|---|
| Standard harmful | AdvBench | Direction | 32 | `a8e6c34d0426c6bb8e3cb0c8d1585530d6bf839dd19cdfbfdba92f99fa039adf` |
| Standard harmless | Alpaca empty-input instructions | Direction | 32 | `a60a2ec783f324f3a2fb2e830a9ed5a20c0714f40130b02fab57682daa6d1c06` |
| Standard harmful | AdvBench | Holdout refusal | 64 | `c7c0d510147b1f38db4465747c1135c298c983a798470e627c3506d95c09af56` |
| Standard harmless | Alpaca empty-input instructions | Holdout KL, coherence, and over-refusal | 64 | `020a483dbc782a1469293dd1e640ee6b033ea7502f819f1fcaee6108cb92f4b4` |
| Matched harmful | Semantic-Harmful pair side | Direction | 32 | `5abd334dfb4d831b7e29c3a917631bc74fd846cc650a1b8f02d391534abead3b` |
| Matched harmless | Semantic-Harmless pair side | Direction | 32 | `5b9a0ae737b99c5accfb5c349fe8874cc40e8f04328d1da224f4d524e9eb01c1` |
| Matched harmful | Semantic-Harmful pair side | Holdout refusal | 64 | `9bd7d33f8db461bbf7751ff647c1153d08aedbe61002ab8a7908b0f9ed8fa81d` |
| Matched harmless | Semantic-Harmless pair side | Holdout over-refusal | 64 | `ea5eb192d028f387d659bfbfb4db6e11b7c36786877973a1197a4b9e5bc594a8` |
| Standard harmful | AdvBench | Final test | 64 | `237940b25adb130e0b525aba268d8fa506785aead4e1e78a6e55c47a55b37d44` |
| Standard harmless | Alpaca | Final test | 64 | `e531a2d800aeff173987345269363c6deea9af9ecb3f7a1929712f639ca8e3f0` |
| Matched harmful | Semantic-Harmful pair side | Final test | 64 | `5fa70dbcc549ac541eda712d07df6032386ddf92e592a397d3f6c1f9e4811048` |
| Matched harmless | Semantic-Harmless pair side | Final test | 64 | `f9bb6e041c99122767dd86cb07cb635edb1d5c9de41b9443dffe0b2d072d5980` |

The direction, candidate-validation, and final-test IDs are disjoint. The matched pairs have mean semantic similarity scores of 0.719 for direction, 0.717 for candidate validation, and 0.725 for final test.

The exact standard row IDs are in `data/splits.toml`. The exact matched pair IDs are in `data/matched_splits.toml`. Both files contain no prompt text.
