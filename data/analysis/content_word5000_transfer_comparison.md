# Quran-MD 5000-Word Content Transfer Comparison

Date: 2026-05-03

## Decision

Do not promote the word-pretrained chunk model.

The larger Quran-MD word experiment succeeded as an auxiliary word recognizer, but failed when transferred directly into the normal chunked content model. The official tuned chunked content baseline stays unchanged.

## Word Dataset

- Source: `Buraaq/quran-md-words`
- Manifest: `data/manifests/hf_quran_md_words_pilot5000_r8.jsonl`
- Rows: `5000`
- Unique normalized word texts: `2571`
- Unique verse IDs: `359`
- Audio duration: `8946.50` seconds
- Cap: maximum `8` clips per normalized word text

## Word-Level Result

| Experiment | Exact | Char accuracy | Edit distance |
| --- | ---: | ---: | ---: |
| Old tuned content model on 5000 word clips | 0.001 | 0.062 | 5.908 |
| New word auxiliary model, held-out words, bp=0.0 | 0.468 | 0.839 | 0.724 |
| New word auxiliary model, held-out words, bp=1.6 | 0.492 | 0.834 | 0.751 |

This is a strong improvement. It proves that real word-level Quran audio is learnable and is much better than approximate ayah chunks.

## Chunk Transfer Result

| Experiment | Exact | Char accuracy | Edit distance | Held-out lexicon coverage |
| --- | ---: | ---: | ---: | ---: |
| Official tuned chunked baseline | 0.700 | 0.892 | 0.731 | 0.000 |
| Word-pretrained chunk model, bp=0.0 | 0.000 | 0.301 | 4.624 | 0.000 |
| Word-pretrained chunk model, bp=1.6 | 0.000 | 0.259 | 5.074 | 0.000 |

The transfer model overfit the training chunks and did not generalize to held-out chunk texts. That means direct pretraining from isolated words into phrase transcription is too abrupt.

## Meaning

We made real progress, but it is not a new official content baseline.

- The word-level path is promising and should continue.
- Direct word-to-chunk transfer is not enough.
- The current official content baseline remains `0.700` exact match and `0.892` character accuracy.

## Next Step

Try a multi-task or curriculum model instead of direct transfer. In practical terms: train the model on word clips and chunk clips together, or use the word recognizer as a separate helper module for segmentation/support. That should preserve the word-learning benefit without destroying chunk generalization.
