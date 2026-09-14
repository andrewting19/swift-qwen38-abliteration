# Prompt Set Report

The exact row numbers are stored in `data/splits.toml`. `scripts/prepare_data.py` retrieves the pinned source files, verifies their hashes, and produces four local JSONL files.

| Set | Count | Median characters | SHA-256 |
|---|---:|---:|---|
| Direction harmful | 32 | 65 | `a8e6c34d0426c6bb8e3cb0c8d1585530d6bf839dd19cdfbfdba92f99fa039adf` |
| Direction harmless | 32 | 58 | `a60a2ec783f324f3a2fb2e830a9ed5a20c0714f40130b02fab57682daa6d1c06` |
| Evaluation harmful | 64 | 72 | `c7c0d510147b1f38db4465747c1135c298c983a798470e627c3506d95c09af56` |
| Evaluation harmless | 64 | 55 | `020a483dbc782a1469293dd1e640ee6b033ea7502f819f1fcaee6108cb92f4b4` |

The direction and evaluation indices do not overlap within either source dataset.

The harmful and harmless groups have similar text lengths. They do not have matched topics or writing styles. A direction measured from them can therefore contain refusal plus some dataset-specific features.

The generated review file is `data/prepared/PROMPT_SETS.md`. It contains every selected prompt and is intentionally excluded from Git.
