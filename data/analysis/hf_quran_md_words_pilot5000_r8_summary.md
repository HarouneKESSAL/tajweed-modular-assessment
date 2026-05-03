# Hugging Face Quran Content Import

- Dataset: `Buraaq/quran-md-words`
- Split: `train`
- Rows written: `5000`
- Rows skipped: `2258`
- Unique normalized texts: `2571`
- Unique reciters: `1`
- Unique verse IDs: `359`
- Total duration seconds: `8946.50`

## Top Reciters

- `quran_md_word_audio`: `5000`

## Top Surahs

- `The Cow`: `4309`
- `The Family of Imraan`: `662`
- `The Opening`: `29`

## Notes

- Audio is stored locally so downstream training does not depend on streaming from Hugging Face.
- Arabic text is normalized by removing diacritics and standardizing common letter variants.
- This import is intentionally capped for pilot experiments; do not download full Quran-scale data blindly.
