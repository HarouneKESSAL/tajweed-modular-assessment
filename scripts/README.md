## Scripts Layout

The flat script layer has been split by responsibility:

- `burst/`: burst-rule training and data builders
- `content/`: content training, evaluation, tuning, and failure analysis
- `data/`: shared manifest/alignment/feature preparation utilities
- `duration/`: duration training, analysis, tuning, and localized/fusion work
- `system/`: end-to-end inference and suite/pipeline evaluation
- `transition/`: transition training, analysis, tuning, and localized work

Root-level `scripts/*.py` files are compatibility wrappers.
They forward to the new module folders so existing commands still work.

Useful entry points:

- `system/run_inference.py`: single-sample modular Tajweed diagnosis
- `system/evaluate_modular_suite.py`: full duration/transition/burst/content scorecard
- `system/generate_demo_report.py`: presentation-ready summary of promoted baselines and experimental findings
- `content/predict_chunked_content.py`: single-chunk content transcription from a manifest row or audio path
- `content/build_subchunk_content_manifest.py`: creates shorter content subchunks
- `content/build_textsplit_train_manifest.py`: creates leakage-safe text-split training manifests
- `content/build_approx_ayah_content_chunks.py`: creates approximate short chunks from ayah-level audio manifests
- `data/audit_clean_expansion_pool.py`: audits clean QuranJSON-matched rows before expanding training data
- `data/import_hf_quran_content.py`: imports capped Hugging Face Quran audio/text pilots into local content manifests
