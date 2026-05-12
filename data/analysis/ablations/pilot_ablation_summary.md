# Pilot Ablation Study

This is a pilot ablation summary. Module contribution rows are **not** true module-off reruns yet; they estimate which modules carry error weight in the verified full-system smoke.

## Full modular baseline

| metric | value |
|---|---:|
| weighted_score | 98.591 |
| duration_acc | 0.993 |
| transition_acc | 0.901 |
| burst_acc | 0.874 |
| chunk_content_char_accuracy | 0.893 |
| chunk_content_exact_match | 0.707 |
| critical_errors | 122 |
| num_errors | 403 |

## Module contribution estimate

| module | current_acc | errors | weighted_sum | weighted_share | severity |
|---|---:|---:|---:|---:|---|
| chunk_content | 0.893 | 122 | 1220.0 | 0.597 | critical |
| burst | 0.874 | 201 | 521.0 | 0.255 | minor/medium |
| transition | 0.901 | 68 | 272.0 | 0.133 | medium |
| duration | 0.993 | 12 | 30.0 | 0.015 | medium/minor |

## Ayah strict batch scoring

| metric | value |
|---|---:|
| samples | 448 |
| avg_score | 76.211 |
| avg_char_accuracy | 0.762 |
| avg_edit_distance | 6.446 |
| exact_rate | 0.062 |
| accepted_rate | 0.062 |
| acceptance_counts | `{'accepted_exact': 28, 'not_accepted': 420}` |
| quality_counts | `{'content_verified_exact': 28, 'likely_same_ayah_but_not_clean': 127, 'same_ayah_candidate_review_required': 170, 'partial_content_match_review_required': 56, 'weak_or_wrong_content': 51, 'almost_correct_review_required': 16}` |

## Expected-text CTC analysis

| metric | value |
|---|---:|
| samples | 448 |
| free_exact_rate | 0.062 |
| expected_text_accepted_rate | 0.114 |
| expected_text_strong_review_rate | 0.404 |
| expected_text_plausible_review_rate | 0.243 |
| avg_free_char_accuracy | 0.762 |
| avg_expected_ctc_loss_per_char | 0.841 |
| avg_expected_ctc_confidence | 0.505 |
| verdict_counts | `{'accepted_free_decode_exact': 28, 'expected_text_strong_but_review_required': 181, 'expected_text_plausible_review_required': 109, 'free_decode_similarity_review_required': 13, 'not_supported': 94, 'accepted_expected_text_near_exact_review_recommended': 23}` |

## Pilot conclusions

- Chunk content carries the largest weighted error contribution because content errors are critical.
- Burst has the largest error count, but lower severity weight than content.
- Duration is already very strong and likely low priority for deeper ablation.
- Transition is worth localizer/threshold ablation because accuracy is good but local support metrics are mixed.
- Ayah strict acceptance is intentionally conservative; expected-text CTC is useful as review evidence, not live acceptance.

## Recommended real ablations next

1. Chunk decoder: tuned blank penalty vs old/default decoder.
2. Transition: whole-verse only vs localizer-assisted.
3. Burst: threshold/localizer variants.
4. Ayah: strict free decode vs expected-text review evidence.
5. Only after these: tiny feature/routing ablations.