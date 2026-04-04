# Folder roles

- `configs/`: experiment and module configs
- `data/`: raw, interim, processed data and manifests
- `scripts/`: CLI entrypoints
- `src/tajweed_assessment/data/`: data loading, labels, collation
- `src/tajweed_assessment/features/`: MFCC, SSL, routing
- `src/tajweed_assessment/models/content/`: ASR/content verification
- `src/tajweed_assessment/models/duration/`: Madd/Ghunnah baseline
- `src/tajweed_assessment/models/transition/`: Idgham/Ikhfa hybrid module
- `src/tajweed_assessment/models/burst/`: Qalqalah CNN
- `src/tajweed_assessment/models/fusion/`: aggregation + feedback
- `src/tajweed_assessment/training/`: loops, metrics, callbacks
- `src/tajweed_assessment/inference/`: end-to-end pipeline
- `tests/`: unit tests
