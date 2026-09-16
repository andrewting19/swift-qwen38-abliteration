# Regularized Edit Plan

Last updated: 2026-09-16

## Reason for this plan

The local direction search succeeded, but the simple full-strength projection
has too much harmless-output drift. The best local rank-2 arm reached 9 of 16
and 11 of 16 HarmBench successes. Its clean harmless KL was 0.2316 nats.

The next test changes the edit rule. It does not repeat direction measurement,
use a public direction, use the final-test split, or create a checkpoint.

## Two controlled changes

### Harmless-mean projection

For each local refusal direction `r`, calculate the masked harmless mean `h` at
the same prompt position and layer. Normalize `h`. Then calculate:

`r_projected = normalize(r - (r dot h) h)`

This removes the part of the measured direction that overlaps the harmless
mean. The method follows the public projected-abliteration proposal:
<https://huggingface.co/blog/grimjim/projected-abliteration>.

Local analysis shows that this is a small change for the two selected vectors:

| Direction | Cosine with harmless mean | Original-to-projected cosine |
| --- | ---: | ---: |
| Position -12, layer 32 | -0.075720 | 0.997129 |
| Position -13, layer 32 | -0.022394 | 0.999749 |

This change is a lower-priority candidate because the direction rotates by only
a small amount.

### Norm-preserving weight projection

For each residual-writing weight, first normalize each output row. Remove the
refusal component from these unit rows. Normalize each edited row again. Restore
the original row length.

This follows the public norm-preserving proposal:
<https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration>.

The local implementation is reversible. Its output hook is mathematically
equivalent to the edited weight. It supports rank 1, rank 2, separate attention
and MLP strengths, biases, and the embedding. A numerical unit test compares
the hook with an explicit edited weight.

## Candidate order

Use one model load. Reuse an existing verified base arm when the prompts and
generation settings match.

Fast screen, 48 tokens:

1. Original position -13, layer-32 rank 1 with norm preservation.
2. Original complementary rank 2 with norm preservation.
3. Projected position -13, layer-32 rank 1 with norm preservation.
4. Projected complementary rank 2 with norm preservation.
5. Original rank 2 with attention alpha 1.0 and MLP alpha 0.9.
6. Original rank 2 with attention alpha 0.9 and MLP alpha 1.0.

Advance at most two arms. Use short-screen refusal markers only as a filter, not
as the final behavior result.

Confirmation, 256 tokens:

1. Reuse the verified 256-token base arm.
2. Generate the two selected candidates.
3. Run local HarmBench once for both candidates.
4. Calculate clean harmless KL, over-refusal, empty outputs, repetition, and
   the secondary strict response label.

## Decision gates

Keep the existing fixed limits:

- Clean harmless mean KL must be at most 0.10 nats.
- Added safe refusal must be at most 5 percentage points.
- Empty and severe repeated outputs must not increase.
- Substantive completion must improve on both harmful validation groups.

The first target is to match or exceed the current local rank-2 HarmBench result
of 9 of 16 and 11 of 16 while passing the KL limit. If no arm does this, do not
use the final-test split and do not create a checkpoint.

## Runtime controls

- The runner can reuse a verified base arm from another compatible screen.
- Candidate reports contain aggregate metadata only.
- Raw harmful data stays in ignored run directories.
- No OpenAI model or remote judge is permitted.
- The public Orca checkpoint remains a diagnostic positive control only.

## Estimated next rental

With the model and judge caches still on the stopped instance, the fast screen
and two confirmations should require about 1.5 to 3.0 GPU hours. At the current
rate of about $1.47 per hour, the expected compute cost is about $2.20 to $4.40.
This is an estimate. Generation throughput is the main uncertainty.

The last observed Vast balance was about $0.57. Do not restart the instance for
this plan until enough credit is available to complete artifact transfer and a
safe stop.
