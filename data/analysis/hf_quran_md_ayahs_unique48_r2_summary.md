# Hugging Face Quran Content Import

- Dataset: `Buraaq/quran-md-ayahs`
- Split: `train`
- Rows written: `96`
- Rows skipped: `1316`
- Unique normalized texts: `48`
- Unique reciters: `2`
- Unique verse IDs: `48`
- Total duration seconds: `1273.06`

## Top Reciters

- `saood_ash_shuraym`: `48`
- `abdurrahmaan_as_sudais`: `48`

## Top Surahs

- `Al-Baqara`: `82`
- `Al-Faatiha`: `14`

## Notes

- Audio is stored locally so downstream training does not depend on streaming from Hugging Face.
- Arabic text is normalized by removing diacritics and standardizing common letter variants.
- This import is intentionally capped for pilot experiments; do not download full Quran-scale data blindly.
