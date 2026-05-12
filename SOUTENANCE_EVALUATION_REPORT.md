# Soutenance Evaluation Report

Project: Tajweed Modular Assessment

Date: 2026-05-09

## 1. Objective

The objective of this project is to build a modular system for Tajweed assessment from Quran recitation audio. Instead of using one black-box model for everything, the system separates the problem into specialized modules:

- Duration rules: madd and ghunnah.
- Transition rules: ikhfa and idgham.
- Burst/articulation rules: qalqalah.
- Content recognition: whether the recited Arabic text matches the expected text.
- Routing and scoring: select the relevant module(s), combine outputs, and produce prioritized feedback.

This modular design is important because Tajweed errors are not all the same. A wrong word, a missing ghunnah, a weak qalqalah, and a transition mistake require different acoustic evidence, different labels, and different feedback.

## 2. Current System Performance After Optimization

The official baseline is the version that passed the most reliable validation gates. Experimental modules are kept separately unless they improve the full system without hidden regressions.

| Component | Main metric | Current result | Status |
|---|---:|---:|---|
| Duration module | Suite accuracy | 0.993 | Promoted |
| Duration module | Strict verse-held-out accuracy | 0.973 | Promoted generalization score |
| Duration: ghunnah | Suite class accuracy | 0.984 | Promoted |
| Duration: madd | Suite class accuracy | 0.995 | Promoted |
| Transition module | Accuracy | 0.901 | Promoted |
| Transition: none | Class accuracy | 0.896 | Promoted |
| Transition: ikhfa | Class accuracy | 0.921 | Promoted |
| Transition: idgham | Class accuracy | 0.857 | Promoted |
| Burst/qalqalah module | Accuracy | 0.874 | Baseline |
| Burst: none | Class accuracy | 0.914 | Baseline |
| Burst: qalqalah | Class accuracy | 0.814 | Baseline |
| Chunked content baseline | Exact match | 0.700 | Official content baseline |
| Chunked content baseline | Character accuracy | 0.892 | Official content baseline |
| Tiny auxiliary content v2 | Exact match | 0.707 | Candidate improvement |
| Tiny auxiliary content v2 | Character accuracy | 0.893 | Candidate improvement |
| Full ayah content recognizer | Free exact rate | 0.063 | Experimental only |
| Full ayah content recognizer | Character accuracy | 0.762 | Experimental only |

Interpretation:

- Duration is the strongest module and generalizes well under verse-held-out evaluation.
- Transition is reliable enough as a promoted specialist module.
- Burst is usable but has room for improvement, especially for missed qalqalah.
- Content improved strongly compared with full-verse recognition, but remains chunk-based.
- Full ayah content recognition is not ready as an official module. It is useful as a scalability experiment.

## 3. Hyperparameter Optimization Summary

The project used controlled manual hyperparameter and threshold optimization. Full Optuna automation is recommended as future work, but the current system already includes several optimized decisions.

| Area | Tuned parameter | Why it matters | Best/current decision |
|---|---|---|---|
| Duration | hardcase weighting | Focuses training on confused madd/ghunnah cases | Used carefully, but only promoted when held-out score improved |
| Duration | localized support thresholds | Checks whether the predicted rule is supported by time-local evidence | Used as evidence/fusion support |
| Transition | class weights | Compensates for imbalance between none, ikhfa, idgham | Used in hardcase transition training |
| Transition | decision thresholds | Balances recall and false positives | Tuned, official single-label transition kept |
| Content | blank penalty | Controls CTC blank dominance and output length | Best open chunk baseline uses blank penalty 1.6 |
| Content | hidden size | Controls model capacity | HD96 chunked model selected as stable baseline |
| Content | tiny auxiliary fine-tuning | Adds small word-level exposure without destroying chunk behavior | Tiny auxiliary v2 improved exact match from 0.700 to 0.707 |
| Learned router | per-label thresholds | Controls module selection sensitivity | Best v5 thresholds: duration 0.30, transition 0.50, burst 0.75 |

Main lesson:

More training or larger data did not always improve the system. Conservative tuning often worked better than aggressive training. This is especially true for the content module, where large full-vocabulary or full-ayah experiments hurt chunk reliability.

## 4. Ablation Study and Value Added by Each Component

Ablation means removing or replacing one part of the system to measure its contribution. The table below summarizes the value added by each component.

| System variant | Observed behavior | Value added |
|---|---|---|
| Full-verse content only | Exact match around 0.016 in early full-verse reference | Not reliable enough alone |
| Chunked content baseline | Exact match 0.700, char accuracy 0.892 | Strong improvement by reducing sequence length and using CTC |
| Tiny auxiliary content v2 | Exact 0.707, char accuracy 0.893 | Small but consistent content improvement |
| Duration module only | Duration suite accuracy 0.993 | Strong detection of madd/ghunnah |
| Transition module only | Transition accuracy 0.901 | Handles ikhfa/idgham family |
| Burst module only | Burst accuracy 0.874 | Adds qalqalah/articulation detection |
| Localized support models | Provide span/evidence support | Improves explainability and confidence checking |
| Learned router candidate | Mixed weak-label exact 0.921 after threshold tuning | Useful research candidate, not official yet |
| Multi-label transition candidate | Gold-only exact around 0.819 | Supports verses with multiple transition labels, but below official transition baseline |
| Severity-aware scoring | Converts errors into weighted penalties | Produces pedagogical priority instead of flat accuracy |

Conclusion:

The largest value comes from the modular decomposition. Each specialist module handles a rule family that would be difficult to explain or debug inside a single end-to-end classifier.

## 5. Supplementary Modules and How They Improved the System

The supplementary modules are the parts beyond the first simple classifiers. Their role is to make the system more robust, explainable, and presentation-ready.

### 5.1 Localized support modules

The localized modules predict whether a Tajweed rule is supported at a time region inside the audio. They help answer:

- Did the model detect the rule only globally?
- Is there local acoustic evidence near the expected position?
- Can we show why a prediction was made?

This is useful for teacher-facing feedback because it moves the system from "wrong label" to "wrong label at this position/time".

### 5.2 Learned fusion and threshold tuning

The project tested learned fusion and threshold tuning for better decision making. The important engineering rule was:

> A new fusion method is promoted only if it improves strict held-out evaluation and does not create hidden regressions.

This prevented overfitting to one metric.

### 5.3 Learned routing

The learned router predicts which modules to activate: duration, transition, or burst. It achieved high performance on the mixed Retasy + weak HF dataset:

- Default exact: 0.895.
- Tuned exact: 0.921.
- Tuned macro-F1: 0.979.

However, because many labels came from weak text patterns, the learned router remains experimental. The official system keeps the rule/metadata router.

### 5.4 Multi-label transition

A verse can theoretically contain multiple transition rule families. The multi-label transition module was added to support this idea. It is an important extension, but the official single-label transition model remains stronger for the current validated data.

### 5.5 Tiny auxiliary content fine-tuning

The most successful content-side improvement was small and conservative:

- Start from the stable chunked content checkpoint.
- Add a tiny auxiliary mix of word samples and chunk samples.
- Use very low learning rate.
- Train only one short stage.

This improved exact match from 0.700 to 0.707 and reduced weighted content errors.

## 6. Comparison With Related Work

Direct comparison is difficult because public Quran learner-error datasets are small and often focus on one surah or a small number of rules. The project therefore compares by scope and design rather than claiming a universal state-of-the-art result.

| Related work or dataset | Focus | Limitation | Our position |
|---|---|---|---|
| RetaSy Quranic audio dataset | Learner recitations from non-Arabic speakers; around 7000 recitations and 1166 annotated samples | Annotated learner errors are limited compared with full Quran scale | Used as the main learner-oriented source |
| QDAT | Tajweed correctness for three rules, around 1500 recordings | Small and rule-limited, often focused on a narrow verse/rule setting | Similar rule-specific motivation, but our system is modular across more components |
| Surah Ikhlas error dataset | Correct/error labels and explanations for Surah Al-Ikhlas, 1506 records | Only one short surah | Useful comparison for learner-error scarcity |
| Quran-MD | Large clean Quran dataset, all ayahs and many reciters, word/ayah audio | Clean/proficient recitation, not learner mistakes | Used for scalability, pretraining, and weak routing/content experiments, not as learner-error ground truth |
| General Quran ASR systems | Recognize Quran recitation text | Usually evaluate ASR, not pedagogical Tajweed feedback | Our system adds rule-specific diagnosis and severity-aware feedback |

Research positioning:

The main contribution is not only raw accuracy. It is a modular assessment architecture with interpretable specialist modules, controlled promotion gates, and error prioritization.

## 7. Error Weighting and Priority Mechanism

The system should not treat all errors equally. A wrong word can change the recitation meaning and is more serious than a small timing weakness. This project therefore uses severity-aware weighting.

This is not a neural attention mechanism. It is better described as a pedagogical priority mechanism.

| Error family | Example | Severity | Reason |
|---|---|---|---|
| Content error | Wrong word or missing word | Critical | Can change meaning and invalidates recitation content |
| Transition error | Missing ikhfa/idgham | Medium | Tajweed rule error, usually khafi/subtle |
| Duration error | Weak madd or ghunnah duration | Minor to medium | Important for Tajweed quality |
| Burst error | Missing or weak qalqalah | Minor to medium | Articulation quality error |

Scoring principle:

```text
final_score = 100 - scaled_weighted_penalty

weighted_penalty = sum(error_count * severity_weight * confidence)
```

Benefits:

- Feedback becomes closer to teacher behavior.
- Critical errors are shown first.
- The system can separate "many small issues" from "one serious issue".
- The final score is more interpretable than raw accuracy alone.

## 8. Scalability

The project is scalable in code and manifest design, but limited by labeled learner-error data.

### 8.1 Data scalability

The manifest system can add more ayahs, reciters, and datasets by creating JSONL rows with:

- audio path
- normalized Arabic text
- rule labels
- expected module route
- metadata such as surah, ayah, reciter, and source

Clean Quran datasets such as Quran-MD can cover all 6236 ayahs, but they mostly contain proficient recitations. This helps with pretraining and weak supervision, but does not replace annotated learner mistakes.

### 8.2 Model scalability

The modular design scales by adding modules rather than retraining one huge model:

- Add a new rule family.
- Build its manifest.
- Train a specialist model.
- Add routing logic.
- Add evaluation and scoring weights.

### 8.3 Current limitation

The project is label-limited, not code-limited.

There are large clean Quran datasets, but large public datasets with many distinct ayahs and real learner mistake labels are rare. This is the main scientific limitation and a valid conclusion for the soutenance.

## 9. Hyperparameter Search and Optuna Future Work

The current project used manual and grid-style tuning. Examples:

- blank penalty sweeps for CTC decoding
- threshold tuning for transition and routing
- hardcase weighting
- hidden dimension experiments
- small vs aggressive auxiliary fine-tuning
- full-verse vs chunked content training

Future work can use Optuna to automate this process.

Example Optuna objective:

```text
objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 96, 128, 160])
    blank_penalty = trial.suggest_float("blank_penalty", 0.0, 2.4)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    train_model(learning_rate, hidden_dim, dropout)
    metrics = evaluate_model(blank_penalty)

    return metrics.exact_match + 0.5 * metrics.char_accuracy
```

Important constraint:

Optimization must be done on validation splits that avoid text/verse leakage. Otherwise, hyperparameter optimization can overfit to repeated Quran phrases.

## 10. Justification of Model Choices

| Choice | Why it was used |
|---|---|
| Modular architecture | Tajweed errors have different acoustic and pedagogical meanings |
| MFCC features | Compact acoustic features suitable for duration, articulation, and rule classifiers |
| Wav2Vec/SSL features | Better speech representations for content recognition |
| BiLSTM encoders | Audio is sequential; past and future context help classify rules |
| CTC loss | Useful when exact frame-level alignment between audio and text is not available |
| Chunked content recognition | Shorter sequences are easier and more reliable than full-verse ASR |
| Localized support models | Add explainability and time-region evidence |
| Severity-aware scoring | Produces teacher-like prioritization of feedback |

Alternative models considered or tested:

- Full-verse content model: too weak for official use.
- Whisper-style content experiments: useful research direction but not promoted.
- Full-vocabulary/ayah content expansions: improved scalability but not enough exact correctness.
- Learned router: promising but not fully gold-validated.
- Multi-label transition: useful future extension, but current single-label transition remains stronger.

## 11. About BERTScore and Certainty Metrics

BERTScore is not the main metric for Quran recitation correctness.

Reason:

Quran recitation assessment needs exact text and phonetic correctness. A semantic similarity metric can say two phrases are close in meaning, but in Quran recitation a small letter change can be a serious error.

Better metrics for this project:

- exact match
- character accuracy
- edit distance
- rule accuracy
- class F1
- CTC expected-text loss
- confidence derived from model probabilities or expected-text CTC support

Where BERTScore could be used:

- evaluating generated natural-language feedback explanations
- comparing feedback text quality, not Quran recitation correctness

Therefore, the project should present CTC confidence and severity-weighted scoring as the main certainty/prioritization tools.

## 12. Corpus Construction Pseudocode

### 12.1 General manifest construction

```text
input:
    raw_audio_files
    metadata
    Quran reference text
    Tajweed rule reference
    optional human labels

for each sample:
    load metadata
    normalize Arabic text
    remove diacritics for model target
    map sample to Quran surah and ayah
    extract rule candidates from reference text
    if human labels exist:
        use human label as trusted target
    else:
        mark label as weak or reference-derived

    build manifest row:
        id
        audio_path
        normalized_text
        surah_id
        ayah_id
        reciter_id
        rule_labels
        route_targets
        source_dataset
        label_confidence_type

write all rows to JSONL manifest
```

### 12.2 Module-specific corpus construction

```text
for each manifest row:
    if row contains madd or ghunnah target:
        add to duration corpus

    if row contains ikhfa or idgham target:
        add to transition corpus

    if row contains qalqalah target:
        add to burst corpus

    if row has expected text and usable audio:
        split or crop into content chunks
        add to content corpus
```

### 12.3 Training and evaluation pipeline

```text
for each module:
    load module corpus
    split into train and validation
    avoid text or verse leakage when possible
    extract audio features
    train model
    tune thresholds on validation
    evaluate:
        accuracy
        class accuracy
        F1
        confusion matrix
        hard examples
    save:
        checkpoint
        metrics JSON
        summary report

run full modular suite:
    route each sample to needed modules
    collect module predictions
    generate diagnostic errors
    apply severity weights
    produce feedback and final score
```

## 13. Marketing and Code-Wise Strengths

This project has several engineering strengths that are easy to explain:

- Reproducible scripts: training, evaluation, confusion analysis, inference, and reporting are separated.
- JSONL manifests: datasets are transparent and easy to audit.
- Modular folders: scripts are grouped by duration, transition, burst, content, routing, and system.
- Unit tests: current test suite passes, which supports code reliability.
- Promotion gates: new models are not trusted just because they trained; they must improve validation metrics.
- Explainable feedback: the output is not only a label, but a diagnosis with rule, position, confidence, and severity.
- Scalable design: new rule families or datasets can be added without rewriting the whole system.

This is the "marketing" part in technical language:

> The project is not only a model. It is an experimental framework for Tajweed assessment, with modular training, controlled evaluation, interpretable feedback, and reproducible reports.

## 14. Final Soutenance Position

The strongest honest conclusion is:

> We built a modular Tajweed assessment prototype. Duration, transition, and burst modules reached usable specialist performance. Content recognition improved significantly when moving from full-verse recognition to chunked CTC recognition, but full learner-level open content assessment remains future work because public learner-error datasets are small. The system is therefore a strong research prototype with clear scalability paths and documented limitations.

What to say if asked whether the system is complete:

> The architecture and evaluation pipeline are complete as a prototype. The main limitation is not the code structure, but the availability of large annotated learner-mistake datasets.

## 15. References

- RetaSy Quranic Audio Dataset: https://huggingface.co/datasets/RetaSy/quranic_audio_dataset
- RetaSy paper: https://arxiv.org/abs/2405.02675
- Quran-MD ayah/word dataset: https://huggingface.co/datasets/Buraaq/quran-md-ayahs
- Quran-MD paper: https://arxiv.org/abs/2601.17880
- QDAT-related DNN/Tajweed work: https://arxiv.org/abs/2503.23470
- Mispronunciation detection with QDAT: https://arxiv.org/abs/2305.06429
- Surah Ikhlas labeled dataset: https://huggingface.co/datasets/MuazAhmad7/Surah_Ikhlas-Labeled_Dataset
- Surah Ikhlas Mendeley dataset: https://data.mendeley.com/datasets/sxtmmr6mvk
