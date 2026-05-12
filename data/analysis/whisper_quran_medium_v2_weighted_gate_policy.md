# Whisper content gate verdict policy

| metric | value |
|---|---:|
| samples | 407 |
| accepted_exact_estimated | 301 |
| accepted_exact_rate | 0.7396 |
| remaining_errors | 106 |
| near_misses_ge_095 | 57 |
| strong_near_misses_ge_098 | 5 |

## Recommended production policy

| verdict | action |
|---|---|
| accepted_exact | auto-pass content gate |
| review_near_exact | review/evidence only, do not auto-pass |
| rejected | stop before Tajweed scoring |

## Why

- Qur’an content should not auto-pass on near matches.
- Near matches are useful for review and debugging.
- Exact/muqattaat-exact remains the only automatic pass condition.

## Error counts

| type | count |
|---|---:|
| single_edit_errors | 72 |
| two_edit_errors | 21 |
| short_predictions_len_delta_le_minus2 | 6 |
| long_predictions_len_delta_ge_2 | 5 |
| muqattaat_remaining_errors | 0 |