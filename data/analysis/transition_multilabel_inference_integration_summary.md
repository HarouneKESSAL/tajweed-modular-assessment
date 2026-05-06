# Multi-label transition inference integration

## Status

The optional multi-label transition predictor is integrated into `run_inference.py`.

Old transition path remains unchanged and still works.

## Profiles

### gold_safe

- ikhfa: 0.40
- idgham: 0.35

Conservative. Avoids over-prediction, but may miss weaker ikhfa examples.

### ikhfa_recall_safe

- ikhfa: 0.25
- idgham: 0.35

More sensitive to ikhfa while keeping idgham conservative.

## Example

Sample:

- retasy_train_002264
- text: من شر ما خلق
- expected transition: ikhfa

Old single-label module:

- predicted ikhfa

Multi-label with gold_safe:

- predicted none
- ikhfa probability: 0.286
- idgham probability: 0.254

Multi-label with ikhfa_recall_safe:

- predicted ikhfa
- ikhfa probability: 0.286
- idgham probability: 0.254

## Decision

The multi-label model should remain optional/experimental for now.

Next step:

Evaluate threshold profiles across the full transition validation set before choosing a default system profile.
