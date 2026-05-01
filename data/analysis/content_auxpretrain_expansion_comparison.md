# Content Auxiliary Pretraining Expansion

Decision: `keep_experimental_do_not_promote_yet`

Reason: The candidate ties baseline exact match and slightly improves char accuracy/edit distance, but does not produce a clear exact-match win. It should remain experimental until the gain is stronger or validated on a broader gate.

## Strict Text-Held-Out Gate

- Baseline exact: `0.700`, char accuracy: `0.892`, edit distance: `0.731`
- Best candidate exact: `0.700`, char accuracy: `0.893`, edit distance: `0.727`
- Delta: exact `+0.000`, char `+0.000`, edit `-0.005`

## New Aligned Chunks

- Baseline char accuracy on correct200 aligned chunks: `0.226`
- Previous conservative r20 char accuracy: `0.231`
- Auxiliary-pretrain candidate char accuracy: `0.239`

## Interpretation

- Auxiliary pretraining helped more than plain conservative mixing.
- The candidate ties exact match and slightly improves char accuracy/edit distance.
- It is still not a clear promotion because exact match did not beat the official baseline.
