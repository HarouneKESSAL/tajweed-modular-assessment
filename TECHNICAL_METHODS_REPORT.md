# Technical Methods Report

This report explains the main technical methods used in the project, the external or pretrained model choices, the feature choices per module, the training and evaluation strategy, and the experiments that were accepted or rejected.

## 1. High-Level Technical Strategy

The system follows a modular design. Instead of training one model to detect every kind of Tajweed error, the implementation separates the problem into acoustically different sub-problems and assigns each to a more appropriate method.

The main technical idea is:

- use a dedicated `content verification` path for transcription-like errors
- use dedicated `rule modules` for Tajweed rule categories
- use `routing` to send only relevant segments or positions to the correct module
- use `aggregation` to reconcile content errors and rule judgments into one diagnosis

This design came directly from the architecture described in [Conceptual_Framework.pdf](C:/Users/anis/Desktop/tajweed-modular-assessment/Conceptual_Framework.pdf).

## 2. Modules and Their Techniques

### 2.1 Content Verification

Purpose:

- detect whether the learner recited the correct content
- identify substitutions, deletions, and insertions relative to canonical text

Main approach used:

- ASR-like content verification with a CTC objective
- later improved by switching from full-verse decoding to chunked content verification

Model family:

- SSL-based acoustic front-end, specifically wav2vec-style representations
- sequence encoder plus CTC decoding

Important implementation idea:

- the content path was first built at full-verse level
- after failure analysis showed strong deletion bias and long-sequence collapse, we moved to chunked content verification

Current best content setup:

- checkpoint: [content_chunked_module.pt](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/content_chunked_module.pt)
- decoder config: [content_chunked_decoder.json](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/content_chunked_decoder.json)
- tuned decoding:
  - lexicon-constrained CTC decoding over the canonical chunk vocabulary
  - blank penalty `1.0`
  - no cleanup

### 2.2 Duration Rules

Purpose:

- detect temporal rules such as `madd` and `ghunnah`

Main approach used:

- sequence-level duration model for rule judgment
- localized duration model as supporting evidence
- learned fusion calibrator to decide when localized evidence should affect the final duration prediction

Model family:

- MFCC-based rule modeling
- temporal sequence modeling
- learned fusion on top of sequence and localized evidence

Important implementation idea:

- the base duration path remained the main rule detector
- local duration evidence was introduced to improve the difficult `ghunnah` cases
- the final promoted duration behavior is not just the raw duration model; it is the approved learned fusion path

Current best duration setup:

- base checkpoint: [duration_module.pt](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/duration_module.pt)
- approved fusion checkpoint: [duration_fusion_calibrator_approved.pt](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/duration_fusion_calibrator_approved.pt)

### 2.3 Transition Rules

Purpose:

- detect assimilation and concealment rules such as `ikhfa` and `idgham`

Main approach used:

- whole-verse transition classifier for the main label decision
- localized transition model for supporting evidence and span-level interpretation
- hard-case mining and retraining for the main transition classifier

Model family:

- sequence classification for clip-level transition labels
- localized span detector for finer evidence

Important implementation idea:

- the localized transition path was useful for support and spans
- but the whole-verse transition classifier remained stronger for the main final decision
- therefore the final transition path is hybrid rather than fully localized

Current best transition setup:

- main checkpoint: [transition_module_hardcase.pt](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/transition_module_hardcase.pt)
- localized support checkpoint: [localized_transition_model.pt](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/localized_transition_model.pt)
- decoder config: [localized_transition_decoder.json](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/localized_transition_decoder.json)

### 2.4 Burst Rules

Purpose:

- detect burst-like articulatory events, specifically `qalqalah`

Main approach used:

- dedicated burst classifier

Model family:

- CNN-based burst detector

Current best burst setup:

- checkpoint: [burst_module.pt](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/burst_module.pt)

## 3. Feature Choices

The project uses different feature representations depending on the module.

### Content Verification Features

- SSL / wav2vec-style representations
- chosen because content verification behaves most like speech recognition
- this follows the PDF’s per-module feature strategy

Relevant PDF reference:

- [Conceptual_Framework.extracted.txt](C:/Users/anis/Desktop/tajweed-modular-assessment/Conceptual_Framework.extracted.txt:488)

### Duration Features

- MFCCs with delta-style temporal information
- chosen because duration tasks depend strongly on frame-level temporal behavior and are cheaper to train than a large SSL end-to-end path

Relevant PDF reference:

- [Conceptual_Framework.extracted.txt](C:/Users/anis/Desktop/tajweed-modular-assessment/Conceptual_Framework.extracted.txt:491)

### Transition and Burst Features

- transition uses the transition rule model plus localized support
- burst uses features appropriate for transient articulatory events

In practice, the code supports multiple feature flows, but the architectural principle remained: feature choice is per-module, not global.

## 4. External or Pretrained Models

The main externally motivated model family used in the project is the SSL / wav2vec-style frontend for content verification.

That means:

- the content path uses a pretrained speech representation family rather than raw handcrafted features alone
- this choice was made because content verification requires more robust phoneme recognition and benefits from transfer learning

The report should present this carefully:

- the project does not rely only on handcrafted audio features
- it uses a hybrid engineering strategy: handcrafted features where they are most appropriate, SSL-based features where they are most appropriate

So the main external model choice you can mention is:

- wav2vec 2.0 style SSL content representation

## 5. Core Engineering Techniques Used During Development

The most important engineering techniques used across the project were:

### 5.1 Hard-Case Mining

Used for:

- transition
- duration
- later tested for chunked content

Purpose:

- identify repeated high-confidence mistakes
- oversample or weight those difficult examples during retraining

Outcome:

- worked well for transition
- failed for chunked content
- was not the final solution for duration

### 5.2 Localized Support Models

Used for:

- transition
- duration

Purpose:

- provide local evidence around rule-bearing spans
- improve interpretability and allow hybrid decisions

Outcome:

- useful as supporting evidence
- not always strong enough to replace the main sequence classifier

### 5.3 Learned Fusion

Used for:

- duration

Purpose:

- combine sequence-level duration predictions with localized duration evidence
- improve `ghunnah` without damaging `madd`

Outcome:

- successful after strict verse-held-out validation
- promoted as the approved duration fusion path

### 5.4 Decoder Tuning

Used for:

- chunked content

Purpose:

- improve CTC decoding without retraining the model

Outcome:

- successful with a blank penalty of `1.0`
- successful again with lexicon-constrained decoding over the chunk vocabulary
- this is the currently promoted content decoder setting

### 5.5 Beam Search Evaluation

Used for:

- chunked content

Purpose:

- test whether a more complex decoder would outperform greedy decoding

Outcome:

- failed badly on the current content model
- beam search is not part of the promoted baseline

### 5.6 Lexicon-Constrained Decoding

Used for:

- chunked content

Purpose:

- restrict decoding to the observed canonical chunk vocabulary
- turn the content task into constrained CTC decoding rather than unconstrained open-string decoding

Outcome:

- clearly improved the chunked content baseline on both the current validation split and the stricter text-held-out split
- promoted as the current content decoder

## 6. Evaluation Strategy

The project did not rely on a single metric or a single test script. Instead, several evaluation layers were built.

### 6.1 Module-Level Evaluation

Each specialist module has targeted evaluators and confusion analysis.

Examples:

- duration confusion analysis
- transition confusion analysis
- localized span evaluation for duration and transition
- content failure mining

### 6.2 Integrated Modular Evaluation

The integrated system is evaluated through:

- [evaluate_modular_pipeline.py](C:/Users/anis/Desktop/tajweed-modular-assessment/scripts/evaluate_modular_pipeline.py)
- [evaluate_modular_suite.py](C:/Users/anis/Desktop/tajweed-modular-assessment/scripts/evaluate_modular_suite.py)

These scripts measure the combined behavior of the modular system, not just isolated models.

### 6.3 Approval Gates

Important changes were not promoted based only on local improvement. They were promoted only after explicit comparison gates.

Most important example:

- learned duration fusion was only promoted after beating the conservative duration baseline on a verse-held-out evaluation

This matters because it shows that the promoted baseline is based on controlled comparisons, not just trial-and-error.

## 7. Accepted Technical Decisions

These are the major methods that were kept in the final baseline.

### Accepted

- modular pipeline architecture
- routing based on content alignment
- specialist rule modules
- diagnosis aggregation
- localized support for duration and transition
- learned duration fusion
- hard-case transition checkpoint
- chunked content verification
- lexicon-constrained CTC decoding with blank penalty for content

## 8. Rejected Technical Decisions

These are methods that were tested and rejected because they degraded the baseline.

### Rejected

- chunked content hard-case oversampling retrain
- larger chunked content model on strict text-held-out split
- raw beam search decoding on the chunked content path

The important presentation point is:

- more complexity did not automatically help
- every major variant was kept or rejected based on measured comparison against the current baseline

## 9. Current Best Technical Baseline

The current best technical baseline is:

- duration:
  - [duration_module.pt](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/duration_module.pt)
  - [duration_fusion_calibrator_approved.pt](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/duration_fusion_calibrator_approved.pt)
- transition:
  - [transition_module_hardcase.pt](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/transition_module_hardcase.pt)
- burst:
  - [burst_module.pt](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/burst_module.pt)
- content:
  - [content_chunked_module.pt](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/content_chunked_module.pt)
  - [content_chunked_decoder.json](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/content_chunked_decoder.json)

## 10. Main Technical Limitation Remaining

The main unresolved technical problem is broader generalization rather than a single failing module.

Why:

- duration has a strong approved baseline
- transition has a strong promoted baseline
- burst is stable
- content is now strong under chunked lexicon-constrained decoding
- but that content gain depends on a closed canonical chunk vocabulary

So the next content work should focus on how much of this performance can be preserved under less constrained decoding or broader vocabulary settings.
