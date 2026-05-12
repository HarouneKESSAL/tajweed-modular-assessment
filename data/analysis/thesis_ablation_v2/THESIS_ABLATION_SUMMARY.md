# Thesis Ablation Summary v2

## Content gate ablation

| system | samples | exact after muqattaat | char accuracy | CER | errors | near misses >= 0.95 | strong near misses >= 0.98 | muqattaat remaining |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Whisper-medium v1 | 407 | 73.46% | 98.03% | 2.05% | 108 | 61 | 6 | 0 |
| Whisper-medium v2 weighted + normfix | 407 | 73.96% | 98.17% | 1.89% | 106 | 57 | 5 | 0 |

## Selected content gate

Whisper-medium v2 weighted + muqattaat normfix is selected because it improves exact acceptance, character accuracy, CER, and remaining error count compared with v1.

## Tajweed module baseline

| module | metric | value |
|---|---|---:|
| duration | accuracy | 99.27% |
| transition | accuracy | 91.01% |
| burst | accuracy | 87.54% |

## Duration within-module ablation: rule breakdown

| class | correct | total | accuracy |
|---|---:|---:|---:|
| ghunnah | 358 | 364 | 98.35% |
| madd | 1276 | 1282 | 99.53% |

## Duration within-module diagnostic: gold support by localized detector

| class | supported | total | support rate |
|---|---:|---:|---:|
| ghunnah | 328 | 364 | 90.11% |
| madd | 1272 | 1282 | 99.22% |

## Duration within-module diagnostic: sequence support by localized detector

| class | supported | total | support rate |
|---|---:|---:|---:|
| ghunnah | 115 | 140 | 82.14% |
| madd | 1492 | 1506 | 99.07% |

## Duration localizer diagnostics

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

## Transition within-module ablation: class breakdown

| class | correct | total | accuracy |
|---|---:|---:|---:|
| none | 381 | 414 | 92.03% |
| ikhfa | 205 | 227 | 90.31% |
| idgham | 42 | 49 | 85.71% |

## Transition within-module diagnostic: gold support by localized detector

| class | supported | total | support rate |
|---|---:|---:|---:|
| idgham | 35 | 39 | 89.74% |
| ikhfa | 77 | 78 | 98.72% |
| none | 0 | 191 | 0.00% |

## Transition within-module diagnostic: whole-verse support by localized detector

| class | supported | total | support rate |
|---|---:|---:|---:|
| idgham | 31 | 35 | 88.57% |
| ikhfa | 73 | 84 | 86.90% |
| none | 0 | 189 | 0.00% |

## Transition localizer diagnostics

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

## Burst within-module ablation: class breakdown at selected threshold

| class | correct | total | accuracy |
|---|---:|---:|---:|
| none | 857 | 958 | 89.46% |
| qalqalah | 541 | 639 | 84.66% |

## Burst threshold ablation

| threshold | rule | accuracy | precision | recall | f1 | FP | FN | none_acc | qalqalah_acc |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| argmax | argmax | 87.41% | 86.38% | 81.38% | 83.80% | 82 | 119 | 91.44% | 81.38% |
| 0.42 | qalqalah_probability_threshold | 86.16% | 79.69% | 87.79% | 83.54% | 143 | 78 | 85.07% | 87.79% |
| 0.43 | qalqalah_probability_threshold | 86.41% | 80.40% | 87.32% | 83.72% | 136 | 81 | 85.80% | 87.32% |
| 0.44 | qalqalah_probability_threshold | 86.54% | 81.36% | 86.07% | 83.65% | 126 | 89 | 86.85% | 86.07% |
| 0.45 | qalqalah_probability_threshold | 86.79% | 82.04% | 85.76% | 83.86% | 120 | 91 | 87.47% | 85.76% |
| 0.46 | qalqalah_probability_threshold | 87.04% | 83.03% | 84.98% | 83.99% | 111 | 96 | 88.41% | 84.98% |
| 0.47 | qalqalah_probability_threshold | 87.54% | 84.27% | 84.66% | 84.47% | 101 | 98 | 89.46% | 84.66% |
| 0.48 | qalqalah_probability_threshold | 87.41% | 84.65% | 83.72% | 84.19% | 97 | 104 | 89.87% | 83.72% |
| 0.49 | qalqalah_probability_threshold | 87.41% | 85.44% | 82.63% | 84.01% | 90 | 111 | 90.61% | 82.63% |
| 0.5 | qalqalah_probability_threshold | 87.41% | 86.38% | 81.38% | 83.80% | 82 | 119 | 91.44% | 81.38% |

## Best burst threshold

- Best accuracy threshold: `0.47` with accuracy 87.54%.
- Best qalqalah F1 threshold: `0.47` with F1 84.47%.

## Legacy chunked CTC content note

The chunked CTC content module is kept only as a legacy ablation baseline. It is not used as the final learner-facing content gate. The final content gate is Whisper-medium v2 weighted + muqattaat normfix.

## Thesis conclusion

The ablation study shows that duration is the strongest Tajweed module, transition is acceptable but idgham remains weaker, and burst/qalqalah is the weakest Tajweed module. The main architectural improvement is replacing the deprecated chunked CTC content module with a learner-style full-ayah Whisper-medium ASR content gate. The selected v2 content gate uses targeted oversampling and muqattaat normalization, and only exact normalized matches are automatically accepted.
