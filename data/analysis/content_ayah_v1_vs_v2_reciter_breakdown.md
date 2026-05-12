# Ayah checkpoint comparison by reciter

- manifest: `data\manifests\content_v6a_short_hf_ayah_r1_hf_ayah_clean_all.jsonl`
- split: `val`
- blank_penalty: 1.2

## Overall

| model | samples | exact | char_accuracy | edit_distance | avg_gold_len | avg_pred_len |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 448 | 0.047 | 0.742 | 7.038 | 27.5 | 25.8 |
| v2 | 448 | 0.062 | 0.762 | 6.446 | 27.5 | 26.0 |

## Reciter deltas

| reciter | samples | v1 char | v2 char | Δ char | v1 edit | v2 edit | Δ edit |
|---|---:|---:|---:|---:|---:|---:|---:|
| minshawy_mujawwad | 12 | 0.547 | 0.637 | +0.090 | 18.250 | 15.000 | -3.250 |
| husary_mujawwad | 19 | 0.787 | 0.828 | +0.041 | 8.211 | 6.632 | -1.579 |
| alafasy | 23 | 0.845 | 0.883 | +0.039 | 5.174 | 4.043 | -1.130 |
| banna | 31 | 0.820 | 0.854 | +0.034 | 6.806 | 5.645 | -1.161 |
| abdurrahmaan_as_sudais | 11 | 0.699 | 0.724 | +0.025 | 11.909 | 11.000 | -0.909 |
| ibrahim_akhdar | 33 | 0.847 | 0.871 | +0.024 | 4.030 | 3.485 | -0.545 |
| abdul_basit_murattal | 46 | 0.847 | 0.871 | +0.024 | 4.891 | 4.304 | -0.587 |
| ali_jaber | 20 | 0.867 | 0.892 | +0.024 | 4.750 | 3.950 | -0.800 |
| abdullaah_3awwaad_al_juhaynee | 41 | 0.296 | 0.319 | +0.023 | 18.024 | 17.561 | -0.463 |
| saood_ash_shuraym | 13 | 0.828 | 0.851 | +0.023 | 7.000 | 6.231 | -0.769 |
| hussary.teacher | 31 | 0.869 | 0.886 | +0.017 | 3.710 | 3.387 | -0.323 |
| ghamadi | 29 | 0.827 | 0.841 | +0.014 | 4.207 | 3.931 | -0.276 |
| warsh_yassin | 31 | 0.802 | 0.814 | +0.012 | 7.387 | 6.903 | -0.484 |
| abdullah_basfar | 11 | 0.837 | 0.848 | +0.011 | 6.273 | 5.909 | -0.364 |
| muhsin_al_qasim | 36 | 0.655 | 0.660 | +0.004 | 5.528 | 5.472 | -0.056 |
| abu_bakr_ash_shaatree | 56 | 0.709 | 0.707 | -0.001 | 4.804 | 4.839 | +0.036 |
| warsh_husary | 5 | 0.579 | 0.547 | -0.032 | 6.200 | 6.800 | +0.600 |

## Biggest v2 improvements

- minshawy_mujawwad: char 0.547 → 0.637 (+0.090), edit 18.25 → 15.00
- husary_mujawwad: char 0.787 → 0.828 (+0.041), edit 8.21 → 6.63
- alafasy: char 0.845 → 0.883 (+0.039), edit 5.17 → 4.04
- banna: char 0.820 → 0.854 (+0.034), edit 6.81 → 5.65
- abdurrahmaan_as_sudais: char 0.699 → 0.724 (+0.025), edit 11.91 → 11.00
- ibrahim_akhdar: char 0.847 → 0.871 (+0.024), edit 4.03 → 3.48
- abdul_basit_murattal: char 0.847 → 0.871 (+0.024), edit 4.89 → 4.30
- ali_jaber: char 0.867 → 0.892 (+0.024), edit 4.75 → 3.95

## Biggest v2 regressions

- warsh_husary: char 0.579 → 0.547 (-0.032), edit 6.20 → 6.80
- abu_bakr_ash_shaatree: char 0.709 → 0.707 (-0.001), edit 4.80 → 4.84
- muhsin_al_qasim: char 0.655 → 0.660 (+0.004), edit 5.53 → 5.47
- abdullah_basfar: char 0.837 → 0.848 (+0.011), edit 6.27 → 5.91
- warsh_yassin: char 0.802 → 0.814 (+0.012), edit 7.39 → 6.90
- ghamadi: char 0.827 → 0.841 (+0.014), edit 4.21 → 3.93
- hussary.teacher: char 0.869 → 0.886 (+0.017), edit 3.71 → 3.39
- saood_ash_shuraym: char 0.828 → 0.851 (+0.023), edit 7.00 → 6.23