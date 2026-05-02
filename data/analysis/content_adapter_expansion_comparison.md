# Content Adapter Expansion

Decision: `do_not_promote`

Reason: Neither adapter variant beats the official chunked baseline. The best adapter result remains below baseline exact match and does not improve new aligned chunks.

Scale note: the final target is the whole Quran, around 6000-7000 verses, so we tested an adapter mechanism that could theoretically add capacity without rewriting the stable baseline.

## Strict Text-Held-Out Gate

- Official baseline exact: `0.700`, char accuracy: `0.892`, edit distance: `0.731`
- Previous best experimental no-adapter exact: `0.700`, char accuracy: `0.893`
- Plain adapter best exact: `0.674`, char accuracy: `0.887`
- Freeze-output adapter best exact: `0.698`, char accuracy: `0.892`

## New Aligned Chunks

- Baseline char accuracy: `0.226`
- Previous no-adapter experimental char accuracy: `0.239`
- Plain adapter char accuracy: `0.228`
- Freeze-output adapter char accuracy: `0.226`

## Interpretation

- The adapter code is reusable, but this adapter experiment did not beat the baseline.
- The official content checkpoint remains unchanged.
- The strongest experimental content result is still the previous auxiliary-pretraining run without adapter.
- For Quran-scale content, the next bottleneck is better aligned/segmented supervised data, not another tiny adapter variation.
