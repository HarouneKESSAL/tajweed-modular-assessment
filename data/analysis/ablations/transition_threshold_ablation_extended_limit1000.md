# Transition threshold ablation

- manifest: `data\manifests\retasy_transition_multilabel_extended.jsonl`
- checkpoint: `transition_module_hardcase.pt`
- limit: 1000

| setting | samples | accuracy |
|---|---:|---:|
| with thresholds | 840 | 0.835 |
| without thresholds / argmax | 840 | 0.852 |

## Class summary: with thresholds

| class | correct | total | accuracy |
|---|---:|---:|---:|
| none | 450 | 564 | 0.798 |
| ikhfa | 209 | 227 | 0.921 |
| idgham | 42 | 49 | 0.857 |

## Class summary: without thresholds

| class | correct | total | accuracy |
|---|---:|---:|---:|
| none | 469 | 564 | 0.832 |
| ikhfa | 205 | 227 | 0.903 |
| idgham | 42 | 49 | 0.857 |