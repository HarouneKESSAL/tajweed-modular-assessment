# Transition threshold ablation

- manifest: `data\manifests\retasy_transition_subset.jsonl`
- checkpoint: `transition_module_hardcase.pt`
- limit: 0

| setting | samples | accuracy |
|---|---:|---:|
| with thresholds | 690 | 0.901 |
| without thresholds / argmax | 690 | 0.910 |

## Class summary: with thresholds

| class | correct | total | accuracy |
|---|---:|---:|---:|
| none | 371 | 414 | 0.896 |
| ikhfa | 209 | 227 | 0.921 |
| idgham | 42 | 49 | 0.857 |

## Class summary: without thresholds

| class | correct | total | accuracy |
|---|---:|---:|---:|
| none | 381 | 414 | 0.920 |
| ikhfa | 205 | 227 | 0.903 |
| idgham | 42 | 49 | 0.857 |