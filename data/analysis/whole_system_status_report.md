# Whole-system status report

This report separates the two correct layers:

1. **Content gate**: learner-style Whisper ASR checks whether the recited ayah content matches.
2. **Tajweed modules**: duration / transition / burst are evaluated only on annotated Tajweed manifests.

## Content gate: Whisper Quran ASR

| metric | value |
|---|---:|
| checkpoint | `C:\Users\anis\Desktop\tajweed-modular-assessment\checkpoints\content_asr_whisper_medium_quran_v1_clean_no_juhaynee` |
| samples | 407 |
| exact_norm_rate | 69.53% |
| exact_compact_rate | 70.76% |
| exact_compact_after_muqattaat_norm | 73.22% |
| avg_char_accuracy | 95.64% |
| CER | 2.47% |
| CER_after_muqattaat_norm | 2.06% |
| WER | 6.93% |
| muqattaat_changed_count | 10 |

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

- Whisper-medium clean no-Juhaynee is the current best learner-style content ASR gate.
- The ASR ayah manifest should not be used directly for Tajweed diagnosis because it lacks rule-level annotations.
- Tajweed module results must come from annotated duration / transition / burst manifests.
- The old chunk-content CTC path is deprecated for the learner content-ASR goal.

## Current recommended architecture

```text
Full ayah audio
→ Whisper-medium Quran ASR content gate
→ Quran normalization + muqattaat normalization
→ if content accepted: run Tajweed modules on annotated Tajweed inputs
→ if content rejected: stop with content mismatch / review required
```