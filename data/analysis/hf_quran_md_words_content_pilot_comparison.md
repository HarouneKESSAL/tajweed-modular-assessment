# Quran-MD Word-Level Content Pilot

Date: 2026-05-03

## Decision

Keep this as experimental. Do not promote it yet.

This pilot is much healthier than the rough ayah-chunk experiment because it uses real word-level audio clips from `Buraaq/quran-md-words`, not guessed timing. The model started to learn: held-out character accuracy improved compared with the old content model on the same word clips. But held-out exact match is still `0.000`, so it is not ready to replace or augment the official content baseline.

## Dataset

- Source: `Buraaq/quran-md-words`
- Manifest: `data/manifests/hf_quran_md_words_pilot512_r4.jsonl`
- Rows: `512`
- Unique normalized word texts: `371`
- Unique verse IDs: `43`
- Audio duration: `859.63` seconds
- Cap: maximum `4` clips per normalized word text

## Results

| Experiment | Split | Samples | Exact | Char accuracy | Edit distance |
| --- | --- | ---: | ---: | ---: | ---: |
| Existing tuned content model on word clips | full | 512 | 0.004 | 0.080 | 5.430 |
| Word pilot model | train words | 410 | 0.120 | 0.480 | 2.532 |
| Word pilot model | held-out word texts | 102 | 0.000 | 0.241 | 3.539 |

## Meaning

This is a promising negative/partial-positive result:

- Negative: the model is not accurate enough yet.
- Positive: real word clips are learnable, unlike the approximate ayah chunks.
- Important: this path moves us away from the small fixed phrase list without hardcoding phrases.

## Next Step

Scale this same experiment to several thousand Quran-MD word clips, then use it as an auxiliary content-learning stage. After that, compare the mixed/distilled content model against the current tuned chunked baseline only.
