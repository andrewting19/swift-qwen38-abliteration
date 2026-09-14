# Dataset Methods Used by Abliteration Projects

## OrcaRouter Qwen3.8

The public model material states that the direction used AdvBench harmful prompts and Alpaca harmless prompts. It does not publish the number of direction prompts, selected row numbers, sample seed, or full massive-activation mask rule.

## Original refusal-direction research

The Arditi repository builds its harmful training pool from AdvBench, MaliciousInstruct, and TDC2023. Its default run samples 128 harmful prompts and 128 harmless prompts from the training pools. It uses Alpaca instructions without extra input as the harmless source. HarmBench is used for harmful validation. JailbreakBench, HarmBench test, and StrongREJECT are used for harmful testing.

## FailSpy Abliterator

- Harmful: `Undi95/orthogonal-activation-steering-TOXIC`.
- Harmless: `tatsu-lab/alpaca`, restricted to rows without extra input.
- Split: scikit-learn 80/20 split with seed 42.

## Sumandora

- Harmful: the original AdvBench harmful-behaviors CSV.
- Harmless: `yahma/alpaca-cleaned`.
- Direction count: 32 randomly sampled prompts from each group. The script does not set a random seed.

## Heretic

- Harmful default: `mlabonne/harmful_behaviors`, first 400 training rows.
- Harmless default: `mlabonne/harmless_alpaca`, first 400 training rows.
- The same two datasets have separate evaluation splits in the default configuration.

## This experiment

We have two independent direction measurements:

1. **Reference direction:** 32 fixed AdvBench rows and 32 fixed Alpaca rows. This uses the same named source families as Orca and matches Sumandora's group size. It is not an exact Orca reproduction because Orca's prompt count, row IDs, and sample rule are not public.
2. **Matched direction:** 32 fixed semantic pairs from `heretic-org/Semantic-Harmful` and `heretic-org/Semantic-Harmless`.

Each direction source has a separate 64-item candidate-validation split and a separate 64-item final-test split. All three roles are disjoint. All revisions, row IDs, and hashes are fixed. Raw harmful text is excluded from Git.
