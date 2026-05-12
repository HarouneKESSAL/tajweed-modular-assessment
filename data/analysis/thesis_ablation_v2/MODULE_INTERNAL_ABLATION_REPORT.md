# Module-internal Ablation Report

This report separates the independent Tajweed modules from the deprecated chunked CTC content module.
Duration, transition, and burst are evaluated from their own annotated manifests.

## Module-level summary

| module | accuracy | interpretation |
|---|---:|---|
| duration | 99.27% | strongest module |
| transition | 91.01% | usable, idgham weaker |
| burst/qalqalah | 87.54% | weakest Tajweed module |

## Duration: within-module class ablation

| rule | correct | total | accuracy |
|---|---:|---:|---:|
| ghunnah | 358 | 364 | 98.35% |
| madd | 1276 | 1282 | 99.53% |

## Duration: localizer support diagnostic

| metric | value |
|---|---:|
| localized_available | 1646 |
| localized_same_as_sequence | 1607 |
| localized_same_rate | 97.63% |
| localized_supports_gold | 1600 |
| localized_supports_gold_rate | 97.21% |
| localized_supports_sequence | 1607 |
| localized_supports_sequence_rate | 97.63% |
| localized_disagrees_with_sequence | 39 |

## Duration: gold support by class

| rule | supported | total | support rate |
|---|---:|---:|---:|
| ghunnah | 328 | 364 | 90.11% |
| madd | 1272 | 1282 | 99.22% |

## Transition: within-module class ablation

| class | correct | total | accuracy |
|---|---:|---:|---:|
| none | 381 | 414 | 92.03% |
| ikhfa | 205 | 227 | 90.31% |
| idgham | 42 | 49 | 85.71% |

## Transition: localizer support diagnostic

| metric | value |
|---|---:|
| localized_available | 308 |
| localized_same_as_whole_verse | 268 |
| localized_same_rate | 87.01% |
| localized_supports_gold | 112 |
| localized_supports_gold_rate | 36.36% |
| localized_supports_whole_verse | 104 |
| localized_supports_whole_verse_rate | 33.77% |
| localized_disagrees_with_whole_verse | 40 |

## Transition: gold support by class

| class | supported | total | support rate |
|---|---:|---:|---:|
| idgham | 35 | 39 | 89.74% |
| ikhfa | 77 | 78 | 98.72% |
| none | 0 | 191 | 0.00% |

## Burst: class ablation at selected threshold

| class | correct | total | accuracy |
|---|---:|---:|---:|
| none | 857 | 958 | 89.46% |
| qalqalah | 541 | 639 | 84.66% |

## Burst: threshold ablation

| threshold | accuracy | precision | recall | f1 | FP | FN | none_acc | qalqalah_acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| argmax | 87.41% | 86.38% | 81.38% | 83.80% | 82 | 119 | 91.44% | 81.38% |
| 0.42 | 86.16% | 79.69% | 87.79% | 83.54% | 143 | 78 | 85.07% | 87.79% |
| 0.43 | 86.41% | 80.40% | 87.32% | 83.72% | 136 | 81 | 85.80% | 87.32% |
| 0.44 | 86.54% | 81.36% | 86.07% | 83.65% | 126 | 89 | 86.85% | 86.07% |
| 0.45 | 86.79% | 82.04% | 85.76% | 83.86% | 120 | 91 | 87.47% | 85.76% |
| 0.46 | 87.04% | 83.03% | 84.98% | 83.99% | 111 | 96 | 88.41% | 84.98% |
| 0.47 | 87.54% | 84.27% | 84.66% | 84.47% | 101 | 98 | 89.46% | 84.66% |
| 0.48 | 87.41% | 84.65% | 83.72% | 84.19% | 97 | 104 | 89.87% | 83.72% |
| 0.49 | 87.41% | 85.44% | 82.63% | 84.01% | 90 | 111 | 90.61% | 82.63% |
| 0.5 | 87.41% | 86.38% | 81.38% | 83.80% | 82 | 119 | 91.44% | 81.38% |

## Module-internal conclusion

Duration is already very strong, with both ghunnah and madd above thesis-ready performance. Transition is acceptable, but idgham is weaker than ikhfa and none. Burst/qalqalah remains the weakest Tajweed module; threshold 0.47 is the best simple calibration, but future improvement likely requires hard-example retraining or better localized windows.
