# Learned Routing v5 Profile Comparison Summary

## Status

Learned routing v5 is now a strong optional routing candidate.

It should not replace the current rule/metadata router yet.

## Official routing path

The official system path remains:

- current rule/metadata router

## Optional learned routing path

Checkpoint:

- `checkpoints/learned_router_v5_retasy_hf_rule_aware_group_text.pt`

Threshold config:

- `configs/learned_router_v5_thresholds.yaml`

Profiles:

- `trusted_retasy_calibrated`
- `weak_policy_tuned`

## Profile roles

### trusted_retasy_calibrated

Best for matching current trusted Retasy routing behavior.

Thresholds:

- use_duration: 0.95
- use_transition: 0.80
- use_burst: 0.70

Recommended use:

- default learned-routing comparison profile
- Retasy-style system analysis
- conservative optional routing comparison

### weak_policy_tuned

Best for broad weak-policy routing.

Thresholds:

- use_duration: 0.30
- use_transition: 0.50
- use_burst: 0.75

Recommended use:

- broad candidate routing
- HF weak-policy coverage
- "do not miss possible modules" analysis

## Comparison results

### Balanced trusted Retasy calibration

Manifest:

- `data/manifests/learned_routing_retasy_calibration_balanced.jsonl`

Samples:

- 520

Results:

| Profile | Exact agreement | Macro F1 vs current |
|---|---:|---:|
| trusted_retasy_calibrated | 0.946 | 0.956 |
| weak_policy_tuned | 0.931 | 0.944 |

Decision:

- `trusted_retasy_calibrated` is better for trusted Retasy calibration.

### All trusted Retasy routing rows

Manifest:

- `data/manifests/learned_routing_dataset_v4_rule_aware_group_text.jsonl`

Samples:

- 1973

Results:

| Profile | Exact agreement | Macro F1 vs current |
|---|---:|---:|
| trusted_retasy_calibrated | 0.975 | 0.976 |
| weak_policy_tuned | 0.972 | 0.970 |

Decision:

- `trusted_retasy_calibrated` best matches current official Retasy routing.
- It almost never misses official modules.
- Most disagreements are extra learned modules.

### Mixed Retasy + HF validation

Manifest:

- `data/manifests/learned_routing_dataset_v5_retasy_hf_rule_aware_group_text.jsonl`

Samples:

- 1642

Results:

| Profile | Exact agreement | Macro F1 vs current |
|---|---:|---:|
| trusted_retasy_calibrated | 0.876 | 0.969 |
| weak_policy_tuned | 0.921 | 0.979 |

Decision:

- `weak_policy_tuned` is better for mixed weak-policy validation.
- This is expected because the mixed validation contains many HF weak-label rows.

## Final decision

Use:

- `trusted_retasy_calibrated` as the default learned-routing comparison profile.
- `weak_policy_tuned` for broad candidate-module analysis.

Do not replace the current official router yet.

## Next step

Add suite-level reporting to compare:

- current official router
- learned v5 trusted_retasy_calibrated
- learned v5 weak_policy_tuned

across duration, transition, burst, and full modular suite evaluations.
