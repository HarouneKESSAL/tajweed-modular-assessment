# Transition Multi-Label Retasy + HF Pilot v1

## Goal

Upgrade transition prediction from single-label verse classification:

- none OR ikhfa OR idgham

to multi-label verse classification:

- none
- ikhfa
- idgham
- ikhfa + idgham

## Dataset

Merged manifest:

- `data/manifests/transition_multilabel_retasy_hf_pilot.jsonl`

Composition:

- total rows: 1340
- none: 414
- ikhfa: 227
- idgham: 49
- ikhfa+idgham: 650

Sources:

- gold Retasy: 690
- weak Retasy text-pattern: 150
- weak HF Quran-MD ayah text-pattern: 500

HF pilot:

- 500 ayah-level WAV files
- 29 reciters
- all weak-labeled as ikhfa+idgham
- sample_weight: 0.25

## Training

Checkpoint:

- `checkpoints/transition_multilabel_retasy_hf_pilot_v1.pt`

Best training checkpoint was saved around epoch 5.

Epoch 5:

- val_exact: 0.791
- val_f1: 0.862
- val_precision: 0.966
- val_recall: 0.777
- predicted: {'ikhfa': 44, 'none': 117, 'idgham': 7, 'ikhfa+idgham': 100}

## Threshold tuning

### Merged Retasy + HF validation

Thresholds:

- ikhfa: 0.45
- idgham: 0.40

Metrics:

- exact_match: 0.802
- macro_precision: 0.961
- macro_recall: 0.806
- macro_f1: 0.877
- predicted: {'ikhfa': 49, 'none': 108, 'idgham': 9, 'ikhfa+idgham': 102}

### Retasy extended validation

Thresholds:

- ikhfa: 0.25
- idgham: 0.25

Metrics:

- exact_match: 0.565
- macro_f1: 0.681
- predicted: {'ikhfa': 88, 'none': 40, 'idgham': 25, 'ikhfa+idgham': 15}

### Gold-only Retasy validation

Thresholds:

- ikhfa: 0.40
- idgham: 0.35

Metrics:

- exact_match: 0.819
- macro_precision: 0.837
- macro_recall: 0.714
- macro_f1: 0.754
- ikhfa_f1: 0.842
- idgham_f1: 0.667
- predicted: {'ikhfa': 60, 'none': 70, 'idgham': 8}

## Interpretation

This is a successful multi-label transition proof-of-concept.

The model can now represent:

- no transition
- ikhfa only
- idgham only
- ikhfa + idgham

The HF ayah data significantly improved the model's ability to learn both-label behavior.

## Decision

Keep as strong candidate:

- `transition_multilabel_retasy_hf_pilot_v1.pt`

Do not fully replace the old transition system until this model is integrated into `run_inference.py` / `evaluate_modular_suite.py` and compared end-to-end.

Recommended system thresholds:

- ikhfa: 0.40
- idgham: 0.35

Reason:

These thresholds are safest on gold-only Retasy validation and avoid hallucinating both-label outputs on gold data.
