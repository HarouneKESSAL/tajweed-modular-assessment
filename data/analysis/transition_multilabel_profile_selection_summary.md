# Multi-label Transition Profile Selection

## Decision

Use `merged_best` as the recommended experimental system profile.

Thresholds:

- ikhfa: 0.45
- idgham: 0.40

## Why

`merged_best` is the best conservative system profile across the profile evaluation runs.

### Gold-only Retasy

- samples: 690
- exact: 0.819
- macro_f1: 0.623
- predicted: {'none': 434, 'ikhfa': 236, 'idgham': 20}

This profile avoids hallucinating `ikhfa+idgham` on gold-only data, where no both-label examples exist.

### Retasy extended

- samples: 840
- exact: 0.677
- macro_f1: 0.502
- predicted: {'none': 529, 'ikhfa': 268, 'idgham': 39, 'ikhfa+idgham': 4}

This is conservative. It under-detects some weak both-label examples but keeps exact match highest.

### Merged Retasy + HF pilot, limit 300

- samples: 300
- exact: 0.813
- macro_f1: 0.624
- predicted: {'none': 186, 'ikhfa': 108, 'idgham': 6}

This was the strongest profile in the quick merged profile evaluation.

### Earlier merged threshold tuning

On the full merged validation split, the same threshold family had:

- thresholds: ikhfa=0.45, idgham=0.40
- exact: 0.802
- macro_f1: 0.877
- predicted: {'ikhfa': 49, 'none': 108, 'idgham': 9, 'ikhfa+idgham': 102}

## Trade-off

`retasy_extended_best` has better macro-F1 on Retasy extended because it is more recall-heavy, but it predicts many more both-label cases and has lower exact match.

For the current system, `merged_best` is safer.

## Status

The old single-label transition module remains the official production path.

The multi-label transition module is now integrated as an optional experimental path and should use `merged_best` by default.
