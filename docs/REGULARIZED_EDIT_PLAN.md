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

## Completed screen

Use one model load. Reuse an existing verified base arm when the prompts and
generation settings match.

The following short arms are complete:

| Arm | Clean harmless KL | Minimum refusal-marker removal |
| --- | ---: | ---: |
| Original rank 2, norm-preserving | 0.2450 | 87.5% |
| Original rank 1, norm-preserving | 0.2981 | 75.0% |
| Source-projected rank 1, norm-preserving | 0.3135 | 75.0% |
| Source-projected rank 2, norm-preserving | 0.2727 | 93.75% |
| Attention 1.0, MLP 0.9 | 0.1965 | 50.0% |
| Attention 0.9, MLP 1.0 | 0.2172 | 68.75% |
| Restore layers 0 through 7 | 0.1968 | 56.25% |

No arm passed the 0.10 KL gate. No arm advances to 256-token confirmation.

## Next candidate

The failed source-projected arms used the harmless mean only at source layer 32.
Target-layer biprojection uses a different harmless mean at every edited layer.

The completed reusable capture contains masked harmless means for positions -12
and -13 at all 64 layers. The prepared candidate has shape `[64, 2, 5120]`.
For each target layer, it:

1. Projects the position -12 direction away from that layer's position -12
   harmless mean.
2. Projects the position -13 direction away from that layer's position -13
   harmless mean.
3. Uses QR to make the two rows orthonormal.

The minimum principal cosine between an original and layer-specific rank-2
subspace is 0.9737. The change is larger than source-layer projection but still
controlled. Run this one candidate next with the reversible layerwise hook.

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

With the model and judge caches still on the stopped instance, the one-candidate
short screen should require about 0.15 to 0.4 GPU hours. If it passes, a
256-token confirmation and local HarmBench pass should require another 0.4 to
0.8 GPU hours. At the current rate of about $1.47 per hour, the expected total
cost is about $0.80 to $1.80. This is an estimate. Generation throughput is the
main uncertainty.

The last observed Vast balance was about $0.086. Do not restart the instance for
this plan until enough credit is available to complete artifact transfer and a
safe stop.
