# Colleague Handoff Report

## 1. Project Goal

The goal of this project was to turn the architecture described in [Conceptual_Framework.pdf](C:/Users/anis/Desktop/tajweed-modular-assessment/Conceptual_Framework.pdf) into a working modular Tajweed assessment system.

The PDF explicitly includes:

- feature extraction
- content verification
- segment routing
- specialist rule modules
- diagnosis aggregation
- feedback generation

So the implementation target was not a single classifier. It was an end-to-end modular system that:

1. reads recitation audio
2. verifies the recited content against canonical text
3. routes relevant segments to the correct Tajweed specialist
4. aggregates the outputs
5. produces structured feedback

## 2. What Has Been Implemented

The current codebase implements the following major subsystems:

### 2.1 Routing and Inference Pipeline

- a central inference pipeline in [pipeline.py](C:/Users/anis/Desktop/tajweed-modular-assessment/src/tajweed_assessment/inference/pipeline.py)
- routing logic that decides whether a segment should be checked by duration, transition, or burst specialists
- hybrid inference paths where localized support models provide evidence alongside main sequence-level classifiers

### 2.2 Rule Modules

- `duration` module for `madd` / `ghunnah`
- `transition` module for `ikhfa` / `idgham`
- `burst` module for `qalqalah`

### 2.3 Content Verification

- an original full-verse content verification path
- a later chunked-content path that became the preferred content baseline

### 2.4 Aggregation and Feedback

- aggregation layer that combines content and rule judgments
- feedback formatting that reports:
  - position
  - rule
  - expected vs predicted
  - confidence
  - localized supporting evidence where available

### 2.5 Evaluation and Analysis Tooling

- module-specific evaluators
- integrated modular suite evaluation
- confusion analyzers
- hard-case miners
- localized span evaluators
- content failure analysis

## 3. Main Development Stages

## 3.1 Stage 1: Build the Modular Pipeline

Initial work focused on making the architecture executable:

- specialist models were wired into a common pipeline
- routing was added
- diagnosis aggregation was implemented
- user-facing feedback/report generation was added

At this stage the system could run end to end, but results were not yet reliable.

## 3.2 Stage 2: Fix Integration Problems

Several early issues came from module interaction rather than model quality.

Examples:

- placeholder content alignment polluted rule-only diagnosis
- some outputs produced meaningless deletion floods
- reporting was too raw to debug effectively

This stage focused on:

- removing broken placeholder behavior
- making error reporting character-aware
- adding confidence values
- adding matched-finding reporting for correct cases

## 3.3 Stage 3: Stabilize the Duration Module

The main duration problem was:

- very strong `madd`
- weak `ghunnah`
- frequent `ghunnah -> madd` confusion

What was tried:

- class weighting
- hard-case duration retraining
- localized duration support
- conservative localizer-based overrides
- learned fusion calibration

What failed:

- simple class weighting helped `ghunnah` but hurt `madd`
- hard-case retraining improved `ghunnah` but degraded total duration accuracy

What worked:

- a learned duration fusion calibrator that combines:
  - sequence duration prediction
  - localized duration evidence
  - contextual features

Important detail:

- this was not promoted immediately
- it first had to beat the conservative baseline on a stricter verse-held-out evaluation

Approved duration result on the verse-held-out gate:

- conservative baseline accuracy: `0.954`
- learned fusion accuracy: `0.973`
- `ghunnah: 0.745 -> 0.872`
- `madd: 0.980 -> 0.985`

## 3.4 Stage 4: Stabilize the Transition Module

The main transition problems were:

- confusion among `none`, `ikhfa`, and `idgham`
- nondeterministic behavior from a dummy SSL projection

What was done:

- fixed deterministic feature behavior first
- built a localized transition model for span evidence
- mined hard transition cases
- retrained the main transition model with hard-case emphasis

What was promoted:

- the hard-case whole-verse transition model as the main classifier
- the localized transition model as support, not replacement

Current transition baseline:

- overall `0.901`
- `none = 0.896`
- `ikhfa = 0.921`
- `idgham = 0.857`

## 3.5 Stage 5: Stabilize the Content Module

This was the hardest subsystem.

### Initial problem

The original full-verse content path was very weak.

Failure analysis showed:

- strong deletion bias
- poor exact match
- long-sequence collapse
- much worse behavior on long phrases

### First successful structural fix

The task was changed from full-verse content decoding to chunked content verification.

That means:

- instead of decoding entire verses at once
- the content model works on short canonical chunks

This immediately improved content performance significantly.

### Decoder-side experiments

What was tried:

- greedy decoding
- blank-penalty tuning
- cleanup/postprocessing
- beam search
- lexicon-constrained decoding

What failed:

- cleanup was not helpful
- raw beam search was much worse than greedy on this model

What improved the baseline:

1. greedy decoding + blank penalty
2. later, lexicon-constrained CTC decoding over the canonical chunk vocabulary

Current content baseline:

- chunked content model
- lexicon-constrained decoder
- blank penalty `1.0`

Current chunked content validation result:

- exact match `0.738`
- char accuracy `0.804`
- edit distance `1.370`

Stricter text-held-out split:

- exact match `0.906`
- char accuracy `0.932`
- edit distance `0.468`

## 4. Problems Encountered

The main problems encountered during development were the following.

### 4.1 Integration problems

- placeholder content behavior interfering with rule-only diagnostics
- routing/reporting interactions producing misleading outputs
- hard-to-read debugging output before confidence-aware reporting was added

### 4.2 Duration-specific problems

- persistent `ghunnah -> madd` confusion
- attempts to improve `ghunnah` often damaged `madd`
- localizer had useful signal but was initially too eager

### 4.3 Transition-specific problems

- feature nondeterminism
- `none -> ikhfa` false positives
- `idgham` recall weakness before hard-case retraining

### 4.4 Content-specific problems

- deletion-heavy CTC behavior
- collapse on longer targets
- chunked hard-case oversampling degraded performance
- larger content model under a stricter split collapsed badly
- beam search decoder failed badly on current logits

### 4.5 Evaluation problems

- some early strong-looking results were suspicious because the split was too easy
- this required stricter approval gates such as verse-held-out evaluation for duration

## 5. Techniques Used

The following techniques were used during development.

### 5.1 Modular specialist modeling

Different Tajweed error families were modeled separately rather than forcing one model to solve everything.

### 5.2 Routing

Routing determines which module should process which segment or sample.

### 5.3 Aggregation

Aggregation reconciles:

- content errors
- rule errors
- hybrid localized evidence

into one diagnosis.

### 5.4 Hard-case mining

Used to identify repeated high-confidence failure types and reweight training.

Worked well for:

- transition

Did not become the final solution for:

- duration
- chunked content

### 5.5 Localized support models

Used for:

- transition
- duration

Purpose:

- add span-level or local evidence
- support or challenge sequence-level predictions
- improve interpretability

### 5.6 Learned fusion

Used for duration.

Purpose:

- decide when localized evidence should override or support the sequence model

Outcome:

- became the approved duration baseline after strict validation

### 5.7 Decoder tuning

Used for chunked content.

Purpose:

- improve text decoding without retraining the core content model

### 5.8 Lexicon-constrained CTC decoding

Used for chunked content.

Purpose:

- restrict decoding to the known canonical chunk vocabulary
- convert noisy frame-level probabilities into better whole-chunk predictions

Outcome:

- clearly improved content baseline over tuned greedy decoding

## 6. External / Pretrained Models and Feature Choices

The project uses a mixed strategy: pretrained SSL features where appropriate, and handcrafted acoustic features where appropriate.

### 6.1 Content verification

Main external model family:

- wav2vec-style SSL features

Used via:

- [ssl.py](C:/Users/anis/Desktop/tajweed-modular-assessment/src/tajweed_assessment/features/ssl.py)

Reason:

- content verification is closest to speech recognition / transcription

### 6.2 Duration, transition, burst

Main acoustic features:

- MFCC-based frame features

Used via:

- [mfcc.py](C:/Users/anis/Desktop/tajweed-modular-assessment/src/tajweed_assessment/features/mfcc.py)

Reason:

- duration and rule modules depend strongly on local temporal/acoustic patterns
- MFCCs were simpler and sufficient for those specialist tasks

### 6.3 Model families in the code

- content:
  - SSL features + CTC content model
- duration:
  - sequence rule model + localized duration model + fusion calibrator
- transition:
  - main transition classifier + localized transition span model
- burst:
  - CNN-based burst classifier

## 7. Accepted vs Rejected Approaches

### Accepted

- modular architecture
- routing
- aggregation
- localized support for duration and transition
- hard-case transition checkpoint
- approved learned duration fusion
- chunked content verification
- lexicon-constrained chunked content decoding

### Rejected

- naive duration hard-case promotion
- chunked content hard-case retrain
- larger chunked content model on stricter split
- raw beam search content decoder
- phrase-specific hardcoding as the main fix strategy

## 8. Current Official Baseline

### Duration

- base checkpoint: [duration_module.pt](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/duration_module.pt)
- approved fusion: [duration_fusion_calibrator_approved.pt](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/duration_fusion_calibrator_approved.pt)

### Transition

- promoted checkpoint: [transition_module_hardcase.pt](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/transition_module_hardcase.pt)

### Burst

- checkpoint: [burst_module.pt](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/burst_module.pt)

### Content

- checkpoint: [content_chunked_module.pt](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/content_chunked_module.pt)
- decoder config: [content_chunked_decoder.json](C:/Users/anis/Desktop/tajweed-modular-assessment/checkpoints/content_chunked_decoder.json)

## 9. Current System Status

The system is now in a much stronger state than it was at the start.

Current status:

- duration: strong approved baseline
- transition: strong promoted baseline
- burst: stable baseline
- content: now materially improved and no longer obviously broken

What still remains open is not “basic functionality.” The open question now is generalization, especially for content beyond the current closed chunk vocabulary setting.

## 10. Short Summary for a Colleague

If the colleague needs the shortest accurate version:

This project implemented the modular Tajweed assessment architecture from the PDF as a working codebase. We built routing, specialist rule modules, aggregation, feedback, and evaluation tooling. Duration was stabilized with a learned fusion model, transition was improved with hard-case retraining plus localized support, and content was rescued by moving from full-verse decoding to chunked content verification and finally a lexicon-constrained CTC decoder. The main problems encountered were module-integration issues, duration `ghunnah` confusion, transition instability, and severe content collapse on longer sequences. The current system now has promoted baselines for duration, transition, burst, and content, with the remaining research issue being generalization beyond the current chunk vocabulary constraints.
