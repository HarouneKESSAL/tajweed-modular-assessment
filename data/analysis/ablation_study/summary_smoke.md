# Modular Ablation Study

Mode: smoke / limited samples

This report compares the official full system against variants where one component is removed or replaced.
A positive delta means the variant is better than the official baseline for that metric, except edit distance where lower is better.

## Summary Table

| Variant | Score | Δ | Duration acc | Δ | Transition acc | Δ | Burst acc | Δ | Content exact | Δ | Content char | Δ | Content edit | Δ better |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official_full_system | 99.198 | +0.000 | 1.000 | +0.000 | 0.925 | +0.000 | 0.900 | +0.000 | 0.925 | +0.000 | 0.981 | +0.000 | 0.075 | +0.000 |
| duration_without_fusion | 98.829 | -0.369 | 0.881 | -0.119 | 0.925 | +0.000 | 0.900 | +0.000 | 0.925 | +0.000 | 0.981 | +0.000 | 0.075 | +0.000 |
| duration_sequence_only | 98.829 | -0.369 | 0.881 | -0.119 | 0.925 | +0.000 | 0.900 | +0.000 | 0.925 | +0.000 | 0.981 | +0.000 | 0.075 | +0.000 |
| transition_original_checkpoint | 98.749 | -0.449 | 1.000 | +0.000 | 0.750 | -0.175 | 0.900 | +0.000 | 0.925 | +0.000 | 0.981 | +0.000 | 0.075 | +0.000 |
| transition_without_thresholds | 99.198 | +0.000 | 1.000 | +0.000 | 0.925 | +0.000 | 0.900 | +0.000 | 0.925 | +0.000 | 0.981 | +0.000 | 0.075 | +0.000 |
| transition_without_localizer | 99.198 | +0.000 | 1.000 | +0.000 | 0.925 | +0.000 | 0.900 | +0.000 | 0.925 | +0.000 | 0.981 | +0.000 | 0.075 | +0.000 |
| content_without_blank_penalty | 99.198 | +0.000 | 1.000 | +0.000 | 0.925 | +0.000 | 0.900 | +0.000 | 0.925 | +0.000 | 0.981 | +0.000 | 0.075 | +0.000 |
| content_tiny_aux_candidate | 99.198 | +0.000 | 1.000 | +0.000 | 0.925 | +0.000 | 0.900 | +0.000 | 0.925 | +0.000 | 0.981 | +0.000 | 0.075 | +0.000 |
| content_original_chunked | 99.679 | +0.481 | 1.000 | +0.000 | 0.925 | +0.000 | 0.900 | +0.000 | 1.000 | +0.075 | 1.000 | +0.019 | 0.000 | +0.075 |

## Variant Details

### official_full_system

Question: What is the current full system performance?

Description: Official promoted configuration with duration fusion, hardcase transition, burst, and tuned chunked content.

Output JSON: `data\analysis\ablation_study\official_full_system.json`

### duration_without_fusion

Question: How much does learned duration fusion add?

Description: Official system but with learned duration fusion disabled.

Output JSON: `data\analysis\ablation_study\duration_without_fusion.json`

### duration_sequence_only

Question: How much do localized duration evidence and fusion add over the sequence model?

Description: Official system but with localized duration support and duration fusion disabled.

Output JSON: `data\analysis\ablation_study\duration_sequence_only.json`

### transition_original_checkpoint

Question: How much did transition hardcase training add?

Description: Official system but using the original transition checkpoint instead of the hardcase checkpoint.

Output JSON: `data\analysis\ablation_study\transition_original_checkpoint.json`

### transition_without_thresholds

Question: How much do transition thresholds add compared with plain argmax?

Description: Official system but transition tuned thresholds are disabled.

Output JSON: `data\analysis\ablation_study\transition_without_thresholds.json`

### transition_without_localizer

Question: Does transition localized evidence affect the reported support analysis?

Description: Official system but localized transition support is disabled.

Output JSON: `data\analysis\ablation_study\transition_without_localizer.json`

### content_without_blank_penalty

Question: How much does the tuned CTC blank penalty add?

Description: Official system but chunked content uses open greedy decoding with blank penalty 1.0.

Output JSON: `data\analysis\ablation_study\content_without_blank_penalty.json`

### content_tiny_aux_candidate

Question: Does the tiny auxiliary content candidate improve the official chunked baseline?

Description: Official system but chunked content uses the tiny auxiliary multitask candidate.

Output JSON: `data\analysis\ablation_study\content_tiny_aux_candidate.json`

### content_original_chunked

Question: How much did the stronger HD96 chunked content baseline add over the earlier chunked model without using a fixed answer list?

Description: Official system but using the first/default chunked content checkpoint with open decoding.

Output JSON: `data\analysis\ablation_study\content_original_chunked.json`

