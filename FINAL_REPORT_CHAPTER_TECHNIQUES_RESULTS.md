# Chapter: Development Methodology, Techniques, Experiments, and Results

## 1. Introduction

This chapter presents the technical development and experimental evaluation of the Tajweed modular assessment system. The objective of the system is to analyze Quran recitation audio and provide diagnostic feedback about several Tajweed-related aspects: duration rules such as madd and ghunnah, transition rules such as ikhfa and idgham, burst/articulation rules such as qalqalah, and content correctness.

From the beginning of the project, the main design choice was to avoid a single monolithic model. Tajweed errors are heterogeneous: some errors concern timing, some concern articulation, some concern transitions between letters, and some concern the actual recited text. For that reason, the project was developed as a modular system composed of specialist models, routing logic, aggregation, and feedback generation.

The final prototype is not a commercial-ready Tajweed tutor, but it is a complete research prototype. It implements the main architecture, evaluates each module independently, compares several experimental variants, and documents the main limitation of the field: large clean Quran recitation datasets exist, but large public datasets with many distinct learner mistakes are still limited.

## 2. Development Evolution From Day 0

The project evolved through several development stages. Each stage solved one limitation discovered in the previous one.

| Stage | Work done | Main observation |
|---|---|---|
| Initial setup | Built base dataset manifests, audio loading, feature extraction, and first model scripts | The project needed a clear data format before reliable training |
| First modular version | Implemented duration, transition, burst, and content modules separately | Modular training made debugging easier than one global classifier |
| Full-verse content attempt | Trained content recognition on complete verses | Full-verse CTC recognition was weak and unstable |
| Chunked content reformulation | Split content into shorter chunks | Content accuracy improved strongly because shorter sequences are easier |
| Rule hardcase mining | Analyzed confusion matrices and mined difficult examples | Targeted hardcases improved transition and duration experiments |
| Localized support models | Added time-region support for duration and transition | Improved interpretability and evidence checking |
| Fusion and threshold tuning | Tuned thresholds, blank penalties, and fusion behavior | Controlled tuning improved reliability without blindly promoting every model |
| Larger clean-data experiments | Tested Quran-MD word/ayah data and weak labels | Clean data is useful for pretraining/scalability, but does not replace learner-error labels |
| Experimental extensions | Learned routing, multi-label transition, full-ayah content scoring | Useful research extensions, but official baseline remains conservative |

This chronology is important because it shows that the final system was not obtained by one training run. It was built through iterative experimentation, error analysis, and controlled promotion gates.

## 3. Datasets and Corpus Construction

The project uses JSONL manifest files as the central corpus format. Each line represents one audio sample or one extracted segment. A manifest row can contain the audio path, normalized Arabic text, Quran metadata, rule labels, reciter information, and routing targets.

In addition to audio datasets, the project used a Quran/Tajweed textual reference extracted from the colored Mushaf idea, locally represented through `external/quranjson-tajwid`. This reference was important because learner audio alone does not contain explicit Tajweed rule positions. The Quran JSON and colored Tajweed files provided structured information about the Quran text, surah and ayah identifiers, and rule-colored spans. These references were transformed into project manifests such as `data/manifests/quranjson_rules.jsonl` and later joined with Retasy samples in `data/manifests/retasy_quranjson_train.jsonl`.

In simple terms, the audio datasets told us what was recited, while the Quran JSON / colored Mushaf reference helped define what rule should exist at each position in the verse. This made it possible to derive targets for duration, transition, burst, routing, and content modules.

The main learner-oriented data source is the RetaSy Quranic audio dataset, which contains crowdsourced recitations from non-Arabic speakers. The published RetaSy paper reports around 7000 recitations from 1287 participants and 1166 annotated recitations across six categories. This makes it useful for learner-oriented experiments, but still limited for detailed Tajweed-rule coverage.

Additional datasets were explored:

- Quran-MD ayah and word audio: useful for large-scale clean recitation, all-ayah coverage, weak routing, and content scalability experiments.
- QDAT: useful as a comparison dataset for three Tajweed rules, but limited in scale and rule coverage.
- Surah Ikhlas labeled dataset: useful for learner-error comparison, but limited to one short surah.

### 3.1 Quran JSON and colored Mushaf reference

The colored Mushaf is a Quran representation where Tajweed rules are visually marked by colors. In the project, this idea was used in structured JSON form through `quranjson-tajwid`. Instead of manually labeling every verse from scratch, the project extracted rule candidates from this reference.

The reference was used for several purposes:

- mapping each sample to a surah and ayah
- keeping a canonical Quran text for comparison
- identifying rule positions in the text
- deriving weak or reference-based labels for rules
- building route targets such as duration, transition, and burst
- validating whether a sample should contain madd, ghunnah, ikhfa, idgham, or qalqalah targets

This step was essential because the machine learning models need supervised targets. For example, if the Quran JSON / colored Tajweed reference indicates that a verse contains a ghunnah position, then the corresponding Retasy audio can be used to train or evaluate the duration module at that expected target position.

The main generated artifacts were:

- `data/manifests/quranjson_rules.jsonl`: extracted Tajweed rule information from the Quran JSON reference
- `data/manifests/retasy_quranjson_train.jsonl`: Retasy samples enriched with Quran JSON metadata and rule references
- module-specific manifests derived from these enriched files

The important limitation is that these labels are reference labels, not always human judgments of the learner's actual pronunciation. Therefore, they define the expected Tajweed rule positions, while the audio model decides whether the acoustic evidence supports correct pronunciation.

### 3.2 Arabic text normalization

Arabic text normalization was necessary because Quran text can contain diacritics and orthographic variants. The preprocessing pipeline removes or standardizes elements such as:

- diacritics
- Quranic annotation signs
- letter variants such as alef forms
- extra whitespace

This makes the target text more stable for CTC training and character-level evaluation.

### 3.3 Corpus construction pseudocode

```text
input:
    audio files
    metadata
    Quran JSON reference text
    colored Mushaf / Tajweed rule reference
    Tajweed rule extraction logic

for each sample:
    load metadata
    load or reference audio path
    normalize Arabic text
    map sample to surah and ayah
    join sample with Quran JSON verse
    extract expected rule targets from colored Tajweed reference
    assign routing labels:
        use_duration
        use_transition
        use_burst
        use_content

    if human learner label exists:
        mark target as trusted
    else:
        mark target as weak or reference-derived

    write JSONL row:
        sample id
        audio path
        normalized text
        rule labels
        rule positions
        route labels
        source dataset
        reciter metadata
```

### 3.4 Module-specific corpus construction

```text
for each manifest row:
    if madd or ghunnah is present:
        add row to duration corpus

    if ikhfa or idgham is present:
        add row to transition corpus

    if qalqalah is present:
        add row to burst corpus

    if expected text is available:
        build content segment or chunk
        add row to content corpus
```

## 4. Techniques Used

### 4.1 MFCC acoustic features

MFCC features were used mainly for rule-oriented modules such as duration, transition, and burst. MFCCs are compact acoustic descriptors that represent the spectral shape of speech. They are suitable for local acoustic phenomena such as duration, articulation, and transitions.

In the project, the MFCC pipeline also uses delta and delta-delta features, which represent short-term changes over time. This helps the model capture dynamic acoustic patterns, not only static frames.

### 4.2 Wav2Vec / SSL features

For content recognition, the project uses self-supervised speech representation features inspired by Wav2Vec-style models. Content recognition is closer to automatic speech recognition, so stronger speech representations are more appropriate than handcrafted MFCCs.

The content model therefore uses SSL features as input and predicts Arabic characters with a CTC objective.

### 4.3 BiLSTM temporal modeling

Several modules use BiLSTM-based sequence modeling. A BiLSTM reads the audio feature sequence in both temporal directions. This is useful because a Tajweed rule can depend on context before and after a letter.

For example, transition rules depend on the relationship between adjacent letters, so temporal context is important.

### 4.4 CTC loss for content recognition

CTC, or Connectionist Temporal Classification, was used for content recognition because the dataset does not always provide exact frame-level alignment between each audio frame and each Arabic character. CTC allows the model to learn from audio and text pairs without requiring precise character timestamps.

The decoder then converts frame-level probabilities into a final text prediction. The project tested greedy decoding, beam search, lexicon-constrained decoding, open decoding, and blank-penalty tuning.

### 4.5 Hardcase mining

Hardcase mining means analyzing mistakes, identifying repeated failure patterns, and giving more attention to difficult examples. This was used especially for duration and transition modules.

The method was:

```text
train baseline model
evaluate model
extract high-confidence mistakes
build hardcase list
retrain or tune with hardcase awareness
compare again against baseline
```

This helped avoid random experimentation. Every new training run was motivated by observed errors.

### 4.6 Localized support modeling

Localized support models estimate whether there is acoustic evidence for a Tajweed rule near its expected position. This does not simply give a global label; it also provides time-related support.

This improves explainability because the system can say not only "the rule is wrong", but also "the expected acoustic evidence was not found near this position".

### 4.7 Threshold tuning and decoder blank penalty

Several modules output probabilities. A threshold controls when a probability becomes a positive prediction. Threshold tuning was used for transition, routing, and localized support.

For CTC content decoding, the blank symbol can dominate the output. A blank penalty was tuned to reduce over-deletion and improve output length. The official chunked content baseline uses a blank penalty of 1.6.

### 4.8 Severity-aware scoring

The system does not treat all errors equally. A wrong word is more serious than a small timing deviation. For that reason, the system uses weighted error penalties.

This is better described as a pedagogical priority mechanism, not a neural attention mechanism. It gives higher importance to critical errors.

```text
weighted_penalty = error_count * severity_weight * confidence
final_score = 100 - scaled_weighted_penalty
```

## 5. Model Choices and Justification

| Model or method | Reason for use |
|---|---|
| Modular architecture | Tajweed errors belong to different acoustic and pedagogical families |
| MFCC + deltas | Efficient for local timing and articulation phenomena |
| Wav2Vec/SSL features | Better for content recognition and speech-like tasks |
| BiLSTM | Captures temporal context before and after a sound |
| CTC | Handles audio-text training without exact character alignment |
| Chunked content model | Short segments are easier than full verses |
| Localized support models | Improve interpretability and evidence checking |
| Severity-aware scoring | Makes feedback closer to teacher priorities |

Alternative directions were also tested:

- Full-verse content recognition: not reliable enough.
- Full-vocabulary content expansion: increased scalability but hurt exact chunk reliability.
- Whisper-style experiments: interesting but not promoted.
- Learned routing: promising, but not official because weak-label validation is not enough.
- Multi-label transition: promising for verses with multiple rule families, but not stronger than the official transition baseline.

## 6. Results Since the Beginning of the Project

### 6.1 Main official results

| Module | Metric | Result |
|---|---:|---:|
| Duration | Suite accuracy | 0.993 |
| Duration | Strict verse-held-out accuracy | 0.973 |
| Duration: ghunnah | Suite class accuracy | 0.984 |
| Duration: madd | Suite class accuracy | 0.995 |
| Transition | Accuracy | 0.901 |
| Transition: none | Class accuracy | 0.896 |
| Transition: ikhfa | Class accuracy | 0.921 |
| Transition: idgham | Class accuracy | 0.857 |
| Burst / qalqalah | Accuracy | 0.874 |
| Content chunked baseline | Exact match | 0.700 |
| Content chunked baseline | Character accuracy | 0.892 |
| Content tiny auxiliary v2 | Exact match | 0.707 |
| Content tiny auxiliary v2 | Character accuracy | 0.893 |

### 6.2 Result evolution by module

| Module | Early result | Improved result | What improved it |
|---|---:|---:|---|
| Duration | Focus accuracy around 0.838 in earlier confusion analysis | Suite accuracy 0.993, verse-held-out 0.973 | Better checkpoint selection, localized evidence, strict validation |
| Transition | Around 0.770 in earlier suite evaluations | 0.901 official accuracy | Hardcase-aware training and thresholding |
| Burst | 0.874 | 0.874 stable baseline | Specialized qalqalah classifier |
| Content full verse | Exact around 0.016, char accuracy around 0.536 | Not promoted | Full-verse CTC was too difficult |
| Content chunked | Exact around 0.383 after first chunked training | 0.700 official open baseline | Chunking, HD96 model, decoder tuning |
| Content tiny auxiliary | 0.700 baseline | 0.707 candidate | Very small word/chunk auxiliary fine-tuning |
| Full ayah content | Free exact 0.063, char 0.762 | Experimental only | Clean ayah data helped char accuracy but not exact reliability |

### 6.3 Important negative results

Negative results are important because they show that the system was not optimized blindly.

| Experiment | Result | Decision |
|---|---|---|
| Full-verse content recognition | Very low exact match | Replaced by chunked content |
| Approximate ayah chunks | Exact match stayed 0.000 | Not promoted |
| Direct word-to-chunk pretraining | Improved word recognition but damaged chunk generalization | Not promoted |
| Full-vocabulary candidate v6b | Candidate much worse than old baseline in side-by-side comparison | Not promoted |
| Learned routing on weak labels | High mixed-label score but weaker Retasy-only validation | Experimental |
| Multi-label transition | Useful extension but below official transition baseline | Experimental |

## 7. Hyperparameter Optimization

The project used manual and grid-style tuning. The main optimized hyperparameters were:

- learning rate
- hidden dimension
- class weights
- hardcase weights
- threshold values
- CTC blank penalty
- decoder type
- number of epochs
- auxiliary data ratio

Examples:

| Component | Hyperparameter | Effect |
|---|---|---|
| Content decoder | blank penalty | Controls deletion/insertion balance |
| Transition classifier | class weights | Reduces bias toward majority class |
| Learned router | label thresholds | Balances false positives and false negatives |
| Content tiny auxiliary | learning rate 0.00001, one epoch | Preserved old behavior while adding small improvement |
| Duration hardcase training | hardcase weights | Improved focus on difficult duration errors when carefully validated |

### 7.1 Optuna as future work

Optuna can automate hyperparameter search in future versions. A possible objective is:

```text
objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 96, 128, 160])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    blank_penalty = trial.suggest_float("blank_penalty", 0.0, 2.4)

    train_model(learning_rate, hidden_dim, dropout)
    metrics = evaluate_model(blank_penalty)

    return metrics.exact_match + 0.5 * metrics.char_accuracy
```

However, Optuna must be used carefully. If the validation split has repeated texts or verse leakage, the optimizer can overfit. For this project, text-held-out and verse-held-out validation are more important than simply maximizing validation accuracy.

## 8. Ablation Study

The ablation study measures the value added by each system component.

| System variant | Performance / behavior | Value added |
|---|---|---|
| Full-verse content only | Exact around 0.016 | Shows full ASR is too difficult with current data |
| Chunked content | Exact 0.700, char 0.892 | Major improvement by simplifying sequence length |
| Chunked + tiny auxiliary v2 | Exact 0.707, char 0.893 | Small but consistent content improvement |
| Duration module | Accuracy 0.993 | Adds reliable madd/ghunnah assessment |
| Transition module | Accuracy 0.901 | Adds ikhfa/idgham assessment |
| Burst module | Accuracy 0.874 | Adds qalqalah assessment |
| Localized support | Time-region evidence | Adds explainability |
| Severity scoring | Weighted penalties | Adds pedagogical prioritization |
| Learned router | Tuned weak-label exact 0.921 | Promising optional routing candidate |
| Multi-label transition | Gold-only exact around 0.819 | Supports future multi-rule transition cases |

Conclusion:

The main value of the system is not a single model score. The value comes from the combination of specialist modules, explainable routing, controlled evaluation, and severity-aware feedback.

### 8.1 Duration module ablation

The duration module was improved in layers. The ablation below explains the contribution of each internal component.

| Duration variant | Result / behavior | Value added |
|---|---|---|
| Basic duration sequence model | Earlier focus accuracy around 0.838 | First usable madd/ghunnah classifier |
| Hardcase-aware duration training | Increased attention to confused ghunnah/madd examples | Helped diagnose imbalance, but aggressive weighting could reduce overall accuracy |
| Localized duration support model | Predicted whether duration evidence exists near expected time positions | Added explainability and local acoustic support |
| Conservative baseline before learned fusion | Verse-held-out accuracy 0.954, ghunnah 0.745, madd 0.980 | Strong but still weaker on ghunnah |
| Learned fusion / calibrated duration path | Verse-held-out accuracy 0.973, ghunnah 0.872, madd 0.985 | Promoted because it improved strict held-out accuracy and reduced errors |

Interpretation:

The main internal improvement was not simply more training. The important gain came from combining sequence-level prediction with localized evidence and validating the fusion on a verse-held-out subset.

### 8.2 Transition module ablation

The transition module also went through several internal variants.

| Transition variant | Result / behavior | Value added |
|---|---|---|
| Early transition classifier | Accuracy around 0.770 in earlier suite runs | First baseline for none/ikhfa/idgham |
| Hardcase-aware transition training | Final official accuracy 0.901 | Improved confusion between none, ikhfa, and idgham |
| Localized transition support | Provided local span evidence for transition rules | Useful for explanation, but not strong enough to replace whole-verse classifier |
| Hybrid transition inference | Used localized evidence as support metadata | Improved interpretability of decisions |
| Multi-label transition classifier | Gold-only exact around 0.819 | Supports future cases where one verse contains more than one transition family |

Interpretation:

The official transition path remains the single-label hardcase-trained classifier because it gives the best validated performance. The multi-label version is useful research work, but not yet stronger than the official baseline.

### 8.3 Burst / qalqalah module ablation

The burst module is simpler than duration and transition because it focuses mainly on qalqalah.

| Burst variant | Result / behavior | Value added |
|---|---|---|
| No burst module | Qalqalah errors cannot be evaluated separately | Missing articulation-specific diagnosis |
| Dedicated burst classifier | Accuracy 0.874 | Adds qalqalah-specific detection |
| Error analysis by class | none accuracy 0.914, qalqalah accuracy 0.814 | Shows that missed qalqalah remains the main improvement target |
| Severity-aware scoring for burst | False negatives treated as missing qalqalah, false positives as weak qalqalah | Converts raw errors into teacher-like feedback priority |

Interpretation:

The burst module adds a rule family that would otherwise be absent from the system. Its value is mainly coverage and feedback specificity.

### 8.4 Content module ablation

The content module had the largest number of experiments because content recognition is the closest part to ASR and was the hardest problem.

| Content variant | Result / behavior | Decision |
|---|---|---|
| Full-verse content model | Exact around 0.016, char around 0.536 | Not reliable enough |
| Chunked content model | Exact 0.700, char 0.892 | Official content baseline |
| Decoder blank-penalty tuning | Best chunked open baseline uses blank penalty 1.6 | Improved CTC output behavior |
| 5000-word auxiliary model | Word-level exact 0.492, char 0.834 on held-out words | Useful for word learning, but separate from chunk baseline |
| Direct word-to-chunk transfer | Chunk exact 0.000, char around 0.301 best | Not promoted |
| Tiny auxiliary content v2 | Exact 0.707, char 0.893 | Small but consistent candidate improvement |
| Full ayah content model | Free exact 0.063, char 0.762 | Experimental scalability path only |
| Full-vocabulary v6b candidate | Side-by-side exact 0.013 vs baseline 0.850 on sampled chunks | Not promoted |

Interpretation:

The strongest content lesson is that shorter chunks are currently much more reliable than full ayahs. Large clean Quran data helps explore scalability, but too much aggressive full-vocabulary training can damage the reliable chunked behavior. The safest improvement was tiny auxiliary fine-tuning, not large retraining.

### 8.5 Routing and scoring ablation

Routing and scoring are not acoustic models, but they strongly affect the final system behavior.

| Variant | Result / behavior | Decision |
|---|---|---|
| Rule/metadata routing | Stable and explainable | Kept as official route selection |
| Learned router v5 | Mixed weak-label exact 0.921, macro-F1 0.979 | Strong candidate but still experimental |
| Retasy-only learned router validation | Much weaker than mixed weak-label validation | Shows risk of source/weak-label bias |
| Unweighted error list | All mistakes appear equally important | Less pedagogically useful |
| Severity-weighted scoring | Critical content errors receive higher priority | Better teacher-like feedback |

Interpretation:

The official router remains conservative because it is explainable and less dependent on weak labels. The learned router is a good research extension, but it needs stronger learner-labeled validation before promotion.

## 9. Error Prioritization and Weighted Scoring

The system classifies errors by severity because not all errors have the same pedagogical importance.

| Error type | Example | Severity | Weighting logic |
|---|---|---|---|
| Wrong word | Incorrect content | Critical | Highest penalty |
| Missing transition rule | Missing ikhfa or idgham | Medium | Important Tajweed error |
| Weak duration | Incorrect madd or ghunnah length | Minor to medium | Depends on rule |
| Weak burst | Missing or weak qalqalah | Minor to medium | Articulation issue |

The weighted scoring system supports feedback prioritization. Instead of listing all errors equally, the system can show the most serious errors first.

This can be described as an error-priority mechanism. It is related in spirit to attention because it focuses the evaluation on important errors, but it is not a neural attention layer.

## 10. Scalability

The codebase is scalable because it is manifest-driven and modular. New datasets can be added by building new JSONL manifests. New Tajweed rule families can be added by creating new specialist modules and connecting them to the routing and scoring layers.

### 10.1 Data scalability

Clean Quran datasets can scale to all 6236 ayahs. Quran-MD, for example, provides large-scale clean ayah and word recitation data. This helped the project explore:

- full-ayah content recognition
- word-level auxiliary learning
- weak routing labels
- scalability beyond the small Retasy subset

However, learner-mistake data is the bottleneck. Public datasets with real learner mistakes are much smaller and less diverse than clean reciter datasets.

### 10.2 Model scalability

Because the system is modular, scaling does not require replacing the whole architecture. Future work can add:

- new rule modules
- larger content recognizers
- better learned routing
- teacher-labeled learner mistakes
- Optuna-based tuning
- word-level or phoneme-level alignment

## 11. Comparison With Related Work

| Work / dataset | Description | Limitation | Relation to this project |
|---|---|---|---|
| RetaSy Quranic Audio Dataset | Crowdsourced Quran recitations from non-Arabic speakers, around 7000 recitations and 1166 annotated samples | Learner labels are limited compared with full Quran scale | Main learner-oriented source |
| QDAT | More than 1500 recordings for three Tajweed rules: madd, ghunnah, and ikhfa | Narrow rule set and limited scale | Similar rule-classification goal, but our architecture is broader |
| Surah Ikhlas labeled dataset | 1506 records labeled by experts for Surah Al-Ikhlas | Only one short surah | Shows scarcity of diverse learner-error datasets |
| Quran-MD | Large clean Quran dataset with ayah/word audio | Clean reciters, not learner mistakes | Used for scalability and weak supervision, not final learner-error validation |
| General Quran ASR work | Focuses on transcription | Often lacks Tajweed-specific diagnostic feedback | Our system adds modular rule diagnosis and weighted feedback |

The project should not claim to outperform all related work. Instead, it should claim a different contribution:

> This work proposes and implements a modular Tajweed assessment framework that combines rule-specific acoustic modules, content verification, routing, diagnostic feedback, and severity-aware scoring.

## 12. About BERTScore and Confidence Metrics

BERTScore is not ideal for Quran recitation correctness. BERTScore measures semantic similarity between texts, but Quran recitation requires exact textual and phonetic correctness. Two strings can be semantically similar but still represent a serious Quran recitation error.

The project therefore uses more appropriate metrics:

- exact match
- character accuracy
- edit distance
- rule accuracy
- F1 score
- CTC loss per character
- model confidence
- expected-text CTC support

BERTScore could be used only for evaluating natural-language feedback quality, not for judging Quran recitation correctness.

## 13. Software Engineering and Marketing Value

From a code and product perspective, the project has several strengths:

- The code is modular and organized by task.
- Scripts are grouped into data, duration, transition, burst, content, routing, and system layers.
- Evaluation outputs are saved as JSON and Markdown reports.
- Unit tests validate important components.
- The system has reproducible command-line workflows.
- Large checkpoints are kept out of Git.
- New models are only promoted after comparison with baselines.
- Feedback is structured and explainable.

This is a strong point for the final report because it shows that the project is not only a notebook experiment. It is a reusable research framework.

## 14. Final Discussion

The final system is ready as a research prototype. Duration, transition, and burst modules reached usable specialist performance. Content recognition improved greatly when the project moved from full-verse recognition to chunked CTC recognition. The system also includes experimental extensions for learned routing, multi-label transition, full-ayah content scoring, and severity-aware evaluation.

The main limitation is data availability. Large clean Quran recitation datasets exist, but large datasets of learner mistakes with detailed Tajweed labels are rare. Therefore, the system should be presented as a modular prototype with strong controlled results, not as a complete commercial correction engine for all Quran recitations.

The most important conclusion is:

> The architecture and evaluation framework are complete enough for a strong prototype. Future performance gains depend mainly on larger and more diverse annotated learner-mistake data.

## 15. References

- RetaSy Quranic Audio Dataset: https://huggingface.co/datasets/RetaSy/quranic_audio_dataset
- RetaSy paper: https://arxiv.org/abs/2405.02675
- Quran-MD ayah dataset: https://huggingface.co/datasets/Buraaq/quran-md-ayahs
- Quran-MD word dataset: https://huggingface.co/datasets/Buraaq/quran-md-words
- QDAT Tajweed rule studies: https://arxiv.org/abs/2503.23470
- Mispronunciation detection with QDAT: https://arxiv.org/abs/2305.06429
- Surah Ikhlas labeled dataset: https://huggingface.co/datasets/MuazAhmad7/Surah_Ikhlas-Labeled_Dataset
- Surah Ikhlas Mendeley dataset: https://data.mendeley.com/datasets/sxtmmr6mvk
