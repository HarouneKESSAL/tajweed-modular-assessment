# Quran-MD Content Pilot Comparison

Date: 2026-05-02

## Decision

Do not promote this Quran-MD content pilot.

The experiment was valuable, but the current content model does not become reliable just by adding ayah-level Quran-MD audio. Full-ayah transcription stayed at `0.000` exact match, and the approximate chunk experiment also stayed at `0.000` exact match. The best interpretation is that the content path needs real word-level clips/timestamps, or a stronger alignment pipeline, before this larger dataset can help.

## Dataset

- Source manifest: `data/manifests/hf_quran_md_ayahs_unique48_r2.jsonl`
- Rows: `96`
- Unique normalized texts: `48`
- Reciters: `abdurrahmaan_as_sudais`, `saood_ash_shuraym`
- Audio duration: `1273.06` seconds

## Approximate Chunk Dataset

- Chunk manifest: `data/manifests/hf_quran_md_ayahs_unique48_r2_approx_chunks.jsonl`
- Rows: `678`
- Method: split each ayah into short word chunks and estimate chunk timing from word character counts.
- Warning: this is only an exploratory bridge. It is not the same as real word-level alignment.

## Results

| Experiment | Samples | Exact | Char accuracy | Edit distance |
| --- | ---: | ---: | ---: | ---: |
| Existing tuned chunked baseline on full ayahs | 96 | 0.000 | 0.111 | 53.719 |
| Aux-pretrained chunk model on full ayahs | 96 | 0.000 | 0.112 | 53.625 |
| Fine-tuned full-ayah pilot, train reciter | 48 | 0.000 | 0.248 | 44.729 |
| Fine-tuned full-ayah pilot, held-out reciter | 48 | 0.000 | 0.117 | 53.500 |
| Approx chunks, old content init, train reciter | 339 | 0.000 | 0.187 | 6.788 |
| Approx chunks, old content init, held-out reciter | 339 | 0.000 | 0.183 | 6.788 |
| Approx chunks, old content init, stronger blank penalty | 339 | 0.000 | 0.175 | 6.909 |
| Approx chunks, scratch model, train reciter | 339 | 0.000 | 0.171 | 7.000 |
| Approx chunks, scratch model, held-out reciter | 339 | 0.000 | 0.134 | 7.239 |

## Meaning

This is a clean negative result. The model is not failing because we forgot to tune a small decoder setting. It is failing because we are asking a small CTC content model to learn long or roughly timed Quran text from weak alignment.

The old chunked baseline remains the official content baseline:

- Exact match: `0.700`
- Character accuracy: `0.892`
- Edit distance: `0.731`

## Next Step

The next best content step is to import or build true word-level content data, then repeat this same reciter-held-out comparison against the tuned chunked baseline. That gives us a fair path away from the small fixed phrase list without trusting noisy approximate timings.
