# Hugging Face Quran Content Import

- Dataset: `Buraaq/quran-md-words`
- Split: `train`
- Rows written: `512`
- Rows skipped: `66`
- Unique normalized texts: `371`
- Unique reciters: `1`
- Unique verse IDs: `43`
- Total duration seconds: `859.63`

## Top Reciters

- `quran_md_word_audio`: `512`

## Top Surahs

- `The Cow`: `483`
- `The Opening`: `29`

## Notes

- Audio is stored locally so downstream training does not depend on streaming from Hugging Face.
- Arabic text is normalized by removing diacritics and standardizing common letter variants.
- This import is intentionally capped for pilot experiments; do not download full Quran-scale data blindly.
