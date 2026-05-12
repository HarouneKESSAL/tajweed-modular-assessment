# Learned Router v5 Retasy + HF Rule-Aware Group-Text

## Dataset

Manifest:

- `data/manifests/learned_routing_dataset_v5_retasy_hf_rule_aware_group_text.jsonl`

Composition:

- rows: 8209
- Retasy trusted rows: 1973
- HF weak all-ayah rows: 6236
- groups: 6062
- group split: text
- group overlap: 0
- features: 69
- rule-aware features: 22

## Training

Checkpoint:

- `checkpoints/learned_router_v5_retasy_hf_rule_aware_group_text.pt`

Best training validation:

- exact: 0.895
- macro_f1: 0.976

## Standalone evaluation after inference feature fix

Validation samples:

- 1642

Metrics at default checkpoint thresholds:

- exact: 0.895
- macro_f1: 0.976

Per-label:

- use_duration: f1=0.969, precision=0.982, recall=0.956
- use_transition: f1=0.988, precision=0.979, recall=0.997
- use_burst: f1=0.970, precision=0.953, recall=0.986

## Threshold tuning

Best thresholds:

- use_duration: 0.30
- use_transition: 0.50
- use_burst: 0.75

Tuned metrics:

- exact: 0.921
- macro_f1: 0.979

Per-label:

- use_duration: f1=0.970, precision=0.975, recall=0.965
- use_transition: f1=0.988, precision=0.979, recall=0.997
- use_burst: f1=0.979, precision=0.987, recall=0.971

## Interpretation

v5 fixes the v3/v4 generalization problem by adding full-ayah HF coverage and rule-aware features.

However, HF labels are weak text-pattern labels. The high score proves the model is consistent with the weak routing policy, not that it is fully gold-validated.

## Decision

Keep v5 as the strongest optional learned-routing candidate.

The current rule/metadata router remains official until end-to-end comparison is complete.
