# Tajweed Modular Assessment

This repository contains a modular Tajweed assessment system. The goal is to analyze a recitation audio sample and evaluate different Tajweed rule families with specialized modules instead of using one large monolithic model.

The system is organized around five main parts:

- **Duration module**: detects and evaluates duration-based rules such as `madd` and `ghunnah`.
- **Transition module**: detects rules around letter transitions such as `ikhfa` and `idgham`.
- **Burst module**: detects burst/articulation rules such as `qalqalah`.
- **Content module**: checks whether the recited content matches the expected Arabic text.
- **System layer**: routes samples to the right modules, combines outputs, evaluates the full pipeline, and produces feedback.

## Repository Structure

```text
tajweed-modular-assessment/
  checkpoints/          Small decoder/threshold config files. Large .pt model weights are ignored by Git.
  configs/              YAML configuration files for data, training, and model hyperparameters.
  data/                 Manifests, alignments, and analysis outputs used by training/evaluation.
  docs/                 Extra documentation or supporting notes.
  external/             External reference datasets/repos, such as Quran/Tajweed rule references.
  notebooks/            Optional notebooks for exploration.
  scripts/              Runnable training, evaluation, analysis, and data-preparation scripts.
  src/                  Main Python package used by the scripts.
  tests/                Unit tests for datasets, models, aggregation, and preprocessing behavior.
  *.md                  Reports, summaries, architecture notes, and presentation/project documentation.
```

## Important Top-Level Files

- `README.md`: this guide.
- `requirements.txt`: Python dependencies needed to run the project.
- `.gitignore`: excludes virtual environments, caches, interim features, and large checkpoint weights.
- `PROGRESS_REPORT_EN.md`: detailed English progress report for teachers.
- `TECHNICAL_METHODS_REPORT.md`: deeper explanation of techniques such as MFCC, wav2vec, CTC, routing, and modular evaluation.
- `RESULTS_SUMMARY.md`: compact summary of current model results.
- `CODEBASE_ARCHITECTURE_NOTES.md`: notes about the current architecture and future cleanup.
- `PRESENTATION_REPORT.md`: report-style presentation text.
- `COLLEAGUE_HANDOFF_REPORT.md`: handoff summary for another developer or teammate.
- `Conceptual_Framework.pdf`: conceptual framework document used as project background.

## `configs/`

This folder contains YAML files that control training and evaluation behavior.

- `configs/data.yaml`: audio/data settings such as sample rate, MFCC settings, and speed normalization.
- `configs/model_duration.yaml`: architecture settings for the duration model.
- `configs/model_content.yaml`: architecture settings for the content model, including wav2vec feature settings and hidden size.
- `configs/model_transition.yaml`: architecture settings for the transition model.
- `configs/model_burst.yaml`: architecture settings for the burst/qalqalah model.
- `configs/train.yaml`: common training settings such as seed, epochs, batch size, learning rate, and device.

## `checkpoints/`

This folder is used for trained models and decoder/threshold settings.

Committed files here are small JSON configs, for example:

- `content_chunked_decoder.json`: closed-set/known-verse content decoder setting.
- `content_chunked_decoder_open.json`: open content decoder setting for phrase-list-independent recognition.
- `content_chunked_decoder_open_hd96.json`: improved open content decoder setting for the larger HD96 content model.
- `localized_duration_decoder.json`: thresholds for localized duration span decoding.
- `localized_transition_decoder.json`: thresholds for localized transition span decoding.
- `transition_thresholds.json`: transition decision thresholds.
- `real_duration_thresholds.json`: thresholds for real-duration classifiers.

Large `.pt` model weights are intentionally ignored by Git because some content checkpoints are around 380 MB. They should be stored separately or with Git LFS if they need to be shared through GitHub.

## `data/`

This folder contains generated project data, not raw application code.

### `data/manifests/`

Manifest files describe audio samples, labels, rule targets, and metadata. They are usually JSONL files where each line is one sample.

Examples:

- `retasy_train.jsonl`: main Retasy-derived training manifest.
- `retasy_quranjson_train.jsonl`: manifest joined with Quran/Tajweed reference labels.
- `retasy_duration_alignment_corpus_torchaudio_strict.jsonl`: strict duration-alignment corpus.
- `retasy_transition_subset.jsonl`: transition-rule subset.
- `retasy_burst_subset.jsonl`: burst/qalqalah subset.
- `retasy_content_chunks.jsonl`: chunked content-recognition dataset.
- `quranjson_rules.jsonl`: extracted Tajweed rule information from Quran JSON references.

### `data/alignment/`

Alignment files map rule targets or characters to approximate audio time positions.

Examples:

- `duration_time_projection_strict.jsonl`: projected duration-rule timing.
- `transition_time_projection_strict.jsonl`: projected transition-rule timing.
- `torchaudio_forced_alignment_strict.jsonl`: torchaudio forced-alignment output.

### `data/analysis/`

Analysis JSON files store evaluation results, confusion matrices, hardcases, suite outputs, and experiment comparisons.

Examples:

- `modular_suite_content_open_hd96_textsplit.json`: full modular suite using the improved open content model.
- `content_open_model_side_comparison.json`: comparison of open content model-side experiments.
- `duration_fusion_verse_holdout.json`: duration fusion held-out evaluation.
- `transition_confusions.json`: transition confusion analysis.
- `chunked_content_lexicon_dependency.json`: analysis of content dependency on phrase-list decoding.
- `final_baseline_results.json`: compact final baseline summary.

## `external/`

This folder contains external reference resources.

- `external/quranjson-tajwid`: Quran/Tajweed reference data used to derive canonical rules and labels.

## `scripts/`

The old flat `scripts/` folder was reorganized by module. Each subfolder contains scripts for one responsibility.

### `scripts/data/`

Shared data-preparation scripts.

- `build_manifests.py`: builds base manifests.
- `extract_features.py`: feature extraction utility.
- `build_torchaudio_alignment_corpus.py`: builds an alignment corpus using torchaudio outputs.
- `run_torchaudio_forced_alignment.py`: runs torchaudio forced alignment.

### `scripts/duration/`

Scripts for duration rules such as `madd` and `ghunnah`.

- `train_duration.py`: trains the main duration model with CTC phoneme output and rule classification.
- `analyze_duration_rule_confusions.py`: prints confusion matrices for duration rule predictions.
- `train_localized_duration_model.py`: trains a localized support model that predicts where duration rules occur in time.
- `evaluate_localized_duration_spans.py`: evaluates localized duration spans.
- `train_duration_fusion_calibrator.py`: trains the learned fusion/calibration layer.
- `mine_duration_hardcases.py`: extracts difficult duration examples for analysis.
- `predict_localized_duration.py`: inspects localized predictions for a sample.

### `scripts/transition/`

Scripts for transition rules such as `ikhfa` and `idgham`.

- `build_transition_manifest.py`: builds the transition-rule dataset.
- `train_transition.py`: trains the whole-clip transition classifier.
- `analyze_transition_confusions.py`: analyzes transition prediction errors.
- `train_localized_transition_model.py`: trains a localized transition support model.
- `evaluate_localized_transition_spans.py`: evaluates localized transition spans.
- `tune_transition_thresholds.py`: tunes transition thresholds.
- `mine_transition_hardcases.py`: extracts hard transition examples.
- `predict_localized_transition.py`: inspects localized transition predictions for a sample.

### `scripts/burst/`

Scripts for burst/articulation rules such as `qalqalah`.

- `build_burst_manifest.py`: builds the burst-rule subset.
- `train_burst.py`: trains the qalqalah/burst classifier.

### `scripts/content/`

Scripts for content-recognition experiments.

- `train_content.py`: trains the older full-verse content model.
- `train_chunked_content.py`: trains the chunked content CTC model.
- `evaluate_chunked_content.py`: evaluates chunked content using greedy, beam, lexicon, or open decoding.
- `predict_chunked_content.py`: predicts text for one chunk from a manifest row or a direct audio path.
- `tune_chunked_content_decoder.py`: tunes decoder settings.
- `analyze_content_failures.py`: analyzes full-verse content failures.
- `analyze_chunked_content_failures.py`: analyzes chunked content failures.
- `mine_chunked_content_hardcases.py`: mines difficult content examples.
- `build_chunked_content_manifest.py`: builds chunk-level content data.

### `scripts/system/`

End-to-end scripts for the full modular system.

- `run_inference.py`: runs inference on one sample and prints the routing plan, diagnosis report, feedback, and matched findings.
- `evaluate_modular_pipeline.py`: evaluates the modular pipeline.
- `evaluate_modular_suite.py`: evaluates duration, transition, burst, and content in one suite.
- `progress_check.ps1`: helper PowerShell script for checking project progress.

### `scripts/README.md`

Short summary of the script-folder layout.

## `src/tajweed_assessment/`

This is the main Python package. The scripts import code from here.

### `src/tajweed_assessment/alignment/`

Utilities for alignment and time projection.

- `prep.py`: prepares alignment inputs.
- `time_projection.py`: projects rule/character positions into time spans.

### `src/tajweed_assessment/data/`

Dataset and manifest utilities.

- `dataset.py`: base dataset logic.
- `audio.py`: audio loading and preprocessing.
- `speed.py`: speed normalization.
- `manifests.py`: manifest loading helpers.
- `quranjson_rules.py`: extracts Tajweed labels from Quran JSON references.
- `localized_duration_dataset.py`: dataset for localized duration training.
- `localized_transition_dataset.py`: dataset for localized transition training.
- `real_duration_dataset.py` and `real_duration_audio_dataset.py`: real-duration classifier datasets.
- `merge_manifest.py`: utilities for merging manifest sources.
- `hf_retasy.py`: helper for Retasy/Hugging Face style data.

### `src/tajweed_assessment/features/`

Feature extraction and routing logic.

- `mfcc.py`: MFCC extraction. MFCCs are compact acoustic features used by duration, transition, and burst models.
- `ssl.py`: self-supervised audio features, especially wav2vec-style features used by content recognition.
- `routing.py`: decides which module should handle a sample based on expected/canonical Tajweed rules.

### `src/tajweed_assessment/models/`

Neural model definitions.

- `models/duration/`: duration-rule model.
- `models/transition/`: transition-rule model.
- `models/burst/`: qalqalah/burst model.
- `models/content/`: wav2vec + CTC content-recognition model.
- `models/common/`: shared model pieces such as BiLSTM encoders, CTC heads, decoding, and losses.
- `models/fusion/`: aggregation, feedback generation, and duration fusion calibration.

### `src/tajweed_assessment/inference/`

- `pipeline.py`: main inference pipeline that runs routed modules and produces structured diagnosis output.

### `src/tajweed_assessment/training/`

Training helpers.

- `engine.py`: generic training/evaluation loop helpers.
- `metrics.py`: token accuracy, sequence accuracy, and decoding metrics.
- `callbacks.py`: checkpointing utilities.

### `src/tajweed_assessment/utils/`

General utilities such as I/O and seeding.

## `tests/`

Unit tests for important behavior.

- `test_dataset.py`: dataset loading and batching behavior.
- `test_models.py`: model forward-pass checks.
- `test_aggregator.py`: aggregation and feedback behavior.
- `test_speed.py`: speed normalization behavior.

Run tests with:

```powershell
.\.venv\Scripts\python -m pytest tests -q
```

## Common Commands

Create and activate the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the full modular suite:

```powershell
.\.venv\Scripts\python scripts\system\evaluate_modular_suite.py --output-json data\analysis\modular_suite.json
```

Run the phrase-list-independent content evaluation:

```powershell
.\.venv\Scripts\python scripts\system\evaluate_modular_suite.py `
  --chunked-content-checkpoint content_chunked_module_hd96_reciter.pt `
  --content-split val `
  --content-split-mode text `
  --content-decoder-config checkpoints\content_chunked_decoder_open_hd96.json `
  --output-json data\analysis\modular_suite_content_open_hd96_textsplit.json
```

Run inference on one sample:

```powershell
.\.venv\Scripts\python scripts\system\run_inference.py --manifest data\manifests\retasy_transition_subset.jsonl --sample-index 1 --show-matches
```

Train/evaluate individual modules:

```powershell
.\.venv\Scripts\python scripts\duration\train_duration.py --manifest data\manifests\retasy_duration_alignment_corpus_torchaudio_strict.jsonl
.\.venv\Scripts\python scripts\transition\train_transition.py --manifest data\manifests\retasy_transition_subset.jsonl
.\.venv\Scripts\python scripts\burst\train_burst.py
.\.venv\Scripts\python scripts\content\train_chunked_content.py
```

Predict content for one chunk:

```powershell
.\.venv\Scripts\python scripts\content\predict_chunked_content.py --sample-index 0
```

Predict content for a direct audio file:

```powershell
.\.venv\Scripts\python scripts\content\predict_chunked_content.py `
  --audio-path data\raw\my_recording.wav `
  --expected-text "الرحمن"
```

## Current Evaluation Notes

Current important results are stored in `data/analysis/`.

Some key artifacts:

- `modular_suite_content_open_hd96_textsplit.json`: open content recognition with no phrase-list coverage.
- `content_open_model_side_comparison.json`: before/after comparison for open content improvements.
- `modular_suite_content_lexicon.json`: known-verse/lexicon-constrained content benchmark.
- `duration_pipeline_verse_holdout_comparison.json`: held-out duration fusion comparison.
- `transition_confusions_hardcase.json`: transition hardcase/confusion analysis.

## Notes For Future Developers

- The project now uses categorized scripts. Prefer `scripts\duration\...`, `scripts\transition\...`, `scripts\content\...`, etc.
- Large trained `.pt` checkpoints are ignored by Git. Share them separately or use Git LFS.
- Decoder JSON files are committed because they are small and define how to reproduce evaluation settings.
- The content module has two evaluation modes:
  - **Known-verse mode**: lexicon decoder, useful when the expected verse/chunk is known.
  - **Open-recognition mode**: greedy CTC decoder, more independent because it does not select from a fixed phrase list.
- The routing module is currently rule/metadata-driven, not a learned audio router.
- The transition module is currently simplified for one main transition class per clip; future work should support multi-label and span-level transition detection.
