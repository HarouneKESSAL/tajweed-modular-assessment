# Content Conservative Expansion R20 Distill

Decision: `do_not_promote`

Reason: The best strict text-held-out score remains below the tuned chunked baseline, and improvement on new aligned chunks is tiny.

## Strict Text-Held-Out Gate

- Baseline exact: `0.700`, char accuracy: `0.892`
- Best candidate exact: `0.695`, char accuracy: `0.891` at blank penalty `1.2`

## New Aligned Chunks

- Baseline char accuracy on correct200 aligned chunks: `0.226`
- Candidate char accuracy on correct200 aligned chunks: `0.231`

## Interpretation

- More clean aligned content alone is not enough yet.
- The candidate preserves old behavior fairly well but does not learn new aligned chunks strongly.
- Next step should change the learning objective/model-side strategy, not just add more rows.
