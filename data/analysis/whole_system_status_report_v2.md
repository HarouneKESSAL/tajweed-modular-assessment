# Whole-system status report

This report separates the two correct layers:

1. **Content gate**: learner-style Whisper ASR checks whether the recited ayah content matches.
2. **Tajweed modules**: duration / transition / burst are evaluated only on annotated Tajweed manifests.

## Content gate: Whisper Quran ASR

| metric | value |
|---|---:|
| checkpoint | `C:\Users\anis\Desktop\tajweed-modular-assessment\checkpoints\content_asr_whisper_medium_quran_v2_weighted` |
| samples | 407 |
| exact_norm_rate | 69.78% |
| exact_compact_rate | 71.74% |
| exact_compact_after_muqattaat_norm | 73.96% |
| avg_char_accuracy | 96.44% |
| CER | 2.15% |
| CER_after_muqattaat_norm | 1.89% |
| WER | 7.16% |
| muqattaat_changed_count | 8 |


## Content gate after muqattaat normalization

| metric | value |
|---|---:|
| accepted_exact_after_muqattaat | 301 / 407 |
| exact_after_rate | 73.96% |
| CER_after | 1.89% |
| remaining_content_errors | 106 |
| muqattaat_remaining_errors | 0 |

## Tajweed module evaluations

| module | samples | units/positions | accuracy |
|---|---:|---:|---:|
| duration | 973 | 1646 | 99.27% |
| transition | 690 | n/a | 91.01% |
| burst | 1597 | n/a | 87.54% |

## Module class summaries

### duration

| class | correct | total | accuracy |
|---|---:|---:|---:|
| ghunnah | 358 | 364 | 98.35% |
| madd | 1276 | 1282 | 99.53% |

### transition

| class | correct | total | accuracy |
|---|---:|---:|---:|
| none | 381 | 414 | 92.03% |
| ikhfa | 205 | 227 | 90.31% |
| idgham | 42 | 49 | 85.71% |

### burst

| class | correct | total | accuracy |
|---|---:|---:|---:|
| none | 857 | 958 | 89.46% |
| qalqalah | 541 | 639 | 84.66% |

## Important note about weighted score

The modular-suite weighted score may still include the old chunk-content CTC module. Do **not** treat that as the final integrated score after the Whisper gate change.

- Old modular-suite estimated score: `98.625`

## Conclusions

- Whisper-medium v2 weighted + muqattaat normfix is the current best learner-style content ASR gate.
- The ASR ayah manifest should not be used directly for Tajweed diagnosis because it lacks rule-level annotations.
- Tajweed module results must come from annotated duration / transition / burst manifests.
- The old chunk-content CTC path is deprecated for the learner content-ASR goal.

## Current recommended architecture

```text
Full ayah audio
→ Whisper-medium v2 Quran ASR content gate
→ Quran normalization + muqattaat normalization
→ if content accepted: run Tajweed modules on annotated Tajweed inputs
→ if content rejected: stop with content mismatch / review required
```