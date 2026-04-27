# Progress Report: Development and Experimentation Status

## 1. Project Objective

The objective of this work was to transform the conceptual architecture proposed in the project framework into a working modular Tajweed assessment system. The goal was not to train one single end-to-end classifier, but to implement a complete pipeline able to process recitation audio, verify the spoken content, route relevant segments to specialized rule modules, aggregate the resulting judgments, and generate interpretable feedback.

At the current stage, the main architecture has been implemented in code and validated through a series of controlled development and experimentation cycles.

## 2. General Development State

The system is now organized around four main functional module families:

- `duration`, for temporal rules such as `madd` and `ghunnah`
- `transition`, for rules such as `ikhfa` and `idgham`
- `burst`, mainly for `qalqalah`
- `content`, for checking whether the expected letters or text were actually recited

In addition to these specialist modules, the project now includes:

- a routing mechanism that directs relevant segments to the correct specialist module
- an aggregation layer that reconciles rule-level and content-level findings
- a feedback layer that converts model outputs into structured diagnostic messages
- batch evaluation and analysis tooling for both module-level and system-level experiments

From a software engineering perspective, the script layer has also been reorganized so that the codebase is now grouped by responsibility:

- `scripts/duration`
- `scripts/transition`
- `scripts/burst`
- `scripts/content`
- `scripts/system`
- `scripts/data`

This makes the project easier to maintain, explain, and extend.

## 3. Techniques and External Models Used

The system does not rely on a single feature extraction method for every task. A key design decision was to use different acoustic representations depending on the nature of the module.

### 3.1 SSL / wav2vec-style features

For the content path, the project uses an SSL-based frontend implemented through a wav2vec-style feature extractor. In code, this is handled through the `Wav2VecFeatureExtractor` in the SSL feature pipeline. When the pretrained torchaudio wav2vec bundle is available, the content model uses those speech representations as the acoustic frontend. When it is not available, the code falls back to a deterministic dummy SSL extractor so that the pipeline remains runnable.

Why this was used:

- content verification is the part of the system that is closest to speech recognition
- SSL features are more suitable than handcrafted low-level features for recovering letters and phoneme-like content
- this is aligned with the conceptual framework, which recommends stronger speech representations for the content verification stage

How it was used:

- for full-verse content verification
- later for chunked content verification
- as the main input representation for the content model trained with a CTC objective

### 3.2 MFCC features

For the temporal and rule-oriented modules, the project uses MFCC-based features, including first and second-order deltas. In practice, the MFCC pipeline extracts:

- base MFCC coefficients
- delta coefficients
- delta-delta coefficients

These are concatenated into a richer frame-level representation.

Why this was used:

- duration and burst problems are highly local and temporal
- MFCC features are computationally efficient
- they work well for frame-level rule modeling where the goal is not full speech transcription

How they were used:

- as the main input to the duration rule model
- as the main input to the burst model
- as the local acoustic representation in transition and localized rule-support models

### 3.3 CTC-based decoding

The content model is trained with a CTC-style objective. This means the model predicts frame-level probability distributions over characters plus a blank symbol, and then a decoder converts those frame-level probabilities into a final text prediction.

Different decoder variants were tested during experimentation:

- greedy decoding
- beam search decoding
- lexicon-constrained decoding

This decoder layer became one of the main research levers for improving the content module.

## 4. Models and Architectural Choices

### 4.1 Content model

The content path is based on an SSL feature frontend plus a CTC-style content verification model. Initially, it was trained on full-verse targets. After failure analysis showed strong deletion bias and long-sequence collapse, the problem was reformulated into chunked content verification, where shorter text-audio pairs were used instead of long full-verse sequences.

This was a major development step because it changed the problem from difficult long-form sequence reconstruction into shorter controlled recognition units.

### 4.2 Duration model

The duration path uses a specialized duration rule model focused on `madd` and `ghunnah`. The main model works at sequence level, but additional localized duration evidence was later added. A learned fusion calibrator was then introduced to decide when the local evidence should affect the final duration prediction.

This means the promoted duration baseline is not just one model checkpoint. It is a combination of:

- a sequence-level duration model
- a localized duration support path
- a learned fusion decision layer

### 4.3 Transition model

The transition path uses a main transition classifier at verse or clip level, along with a localized transition support model. The whole-verse classifier remains the main decision-maker, while the localized path contributes extra evidence and span-level interpretability.

This design was chosen because the localized transition path was useful for interpretation, but not strong enough to replace the main transition classifier.

### 4.4 Burst model

The burst path uses a dedicated burst classifier, mainly for `qalqalah`. It is a more specialized module and reached stability faster than the others.

## 5. How Each Module Was Built

This section explains the implementation process for each module separately, focusing on how the modules were constructed in code.

### 5.1 How the Duration Module Was Built

The duration module was built around the idea that temporal Tajweed rules should be modeled from frame-level acoustic information rather than from full speech-recognition style decoding. The first implementation step was to define a duration-specific dataset structure that links each recitation sample to canonical duration labels such as `madd` and `ghunnah`. After that, MFCC-based features were extracted from the audio because duration judgments depend mainly on local temporal behavior and are well represented by frame-level cepstral features.

The main duration model was then implemented as a sequence model that produces:

- phoneme-related sequence information
- rule-level duration judgments

Once the baseline was running, additional tooling was built around it:

- duration confusion analysis
- duration hard-case mining
- localized duration datasets and models
- duration disagreement analysis between the main model and the localizer

When it became clear that the main difficulty was the `ghunnah` versus `madd` distinction, a localized duration support model was added. After that, a learned fusion calibrator was implemented to combine:

- sequence-level duration predictions
- localized duration confidence
- context information around the current character

So the duration module was built in layers:

1. dataset and labels
2. MFCC feature pipeline
3. sequence duration model
4. localized support model
5. learned fusion layer
6. strict held-out approval evaluation

### 5.2 How the Transition Module Was Built

The transition module was built to classify rules such as `ikhfa` and `idgham` at utterance or segment level. The first step was to construct transition-specific manifests that associate recitation samples with canonical transition labels. The acoustic side combined MFCC features with a lightweight SSL-support path, because transition behavior is local but also benefits from richer contextual representations.

The initial transition classifier was implemented as a whole-verse or clip-level model. Once that baseline existed, detailed confusion analysis was added to inspect where the model confused:

- `none`
- `ikhfa`
- `idgham`

This led to two major extensions:

- a localized transition model for span-level support
- a hard-case mining and retraining pipeline for the main transition classifier

The localized transition path was implemented to answer a different question from the whole-verse classifier. The whole-verse classifier answers which transition rule is present overall, while the localized model tries to indicate where the evidence is in time. Since the localized model was useful but not strong enough to replace the main classifier, the final module architecture became hybrid:

- whole-verse classifier for the final transition decision
- localized model for supporting evidence and span interpretation

So the transition module was built in layers:

1. transition manifests and labels
2. clip-level transition classifier
3. deterministic feature correction
4. confusion analysis
5. localized transition model
6. hard-case retraining
7. hybrid inference path

### 5.3 How the Burst Module Was Built

The burst module was built as a dedicated specialist for `qalqalah`, which is acoustically different from duration and transition rules. Since this is a more localized burst-like phenomenon, the design was simpler from the beginning. A burst-specific manifest was created, MFCC-style local acoustic features were used, and a dedicated classifier was trained directly for burst detection.

Compared with the other modules, the burst module required fewer redesign cycles. It did not need the same level of hybridization or calibration because the baseline behavior was already relatively stable once the data and feature pipeline were in place.

So the burst module was built through:

1. burst manifest construction
2. local acoustic feature extraction
3. dedicated burst classifier training
4. batch evaluation and integration into the system suite

### 5.4 How the Content Module Was Built

The content module was built differently from the rule modules because its purpose is closer to speech recognition. The first implementation used an SSL frontend with a CTC-based content model at full-verse level. The idea was to compare the spoken recitation against the expected canonical text and detect content-level errors such as deletions, substitutions, and insertions.

After the first evaluation stage, failure analysis showed that the full-verse design was too weak. The model had strong deletion bias and often collapsed on longer sequences. Because of that, the content path was redesigned rather than simply retrained.

The next implementation step was to build a chunked content pipeline:

- create chunked content manifests
- align shorter text segments with shorter audio regions
- train a chunk-level content verification model instead of a full-verse model

Once chunking was in place, the content work shifted to decoding and evaluation methodology. Several decoding and training variants were implemented and tested:

- greedy decoding
- beam search decoding
- decoder blank-penalty tuning
- hard-case chunked retraining
- larger chunked model on a stricter split
- lexicon-constrained decoding

The best-performing version became the chunked content model combined with lexicon-constrained CTC decoding. In that design:

- the model outputs frame-level character probabilities
- the decoder chooses the best valid chunk from the canonical chunk vocabulary

So the content module was built in layers:

1. full-verse content baseline
2. content failure analysis
3. chunked content dataset and training path
4. decoder tuning
5. lexicon-constrained decoding
6. strict split evaluation and comparison

### 5.5 How the System-Level Module Was Built

Beyond the specialist models, an important part of the implementation was the system-level layer. This includes:

- routing logic
- diagnosis aggregation
- feedback generation
- modular suite evaluation

Routing was built so the system can decide which specialist module should examine which kind of sample or segment. Aggregation was built so that content findings and rule findings can be combined consistently instead of being reported independently. Feedback generation was then built on top of the aggregated diagnostics so that the system can return human-readable outputs instead of raw tensors or labels.

This system-level layer is what transforms separate trained models into a real modular assessment pipeline.

## 6. Development and Experimentation by Module

### 6.1 Duration Module

The duration module was one of the first major modules to stabilize. Early versions performed strongly on `madd` but were substantially weaker on `ghunnah`. Several strategies were tested to improve this behavior:

- class weighting
- hard-case mining
- localized duration support
- conservative fusion rules
- learned fusion calibration

Some of these helped only partially. For example, simple hard-case retraining improved `ghunnah` in some runs but damaged `madd`, which made the overall baseline worse. The final promoted version uses:

- the main duration model for sequence-level prediction
- localized duration evidence for supporting information
- an approved learned fusion calibrator that decides when localized evidence should influence the final judgment

This version was not promoted immediately. It was first validated on a stricter verse-held-out split to reduce the risk of overfitting to repeated contexts.

Strict duration validation result:

- conservative baseline accuracy: `0.954`
- learned fusion accuracy: `0.973`
- `ghunnah`: `0.745 -> 0.872`
- `madd`: `0.980 -> 0.985`

This means the duration module can now be considered a strong component of the baseline system.

### 6.2 Transition Module

The transition module initially suffered from instability and confusion between `none`, `ikhfa`, and `idgham`. The first major correction was to remove nondeterministic behavior in the feature path so that repeated evaluations became reproducible. After that, confusion analysis and hard-case mining were used to focus retraining on the most difficult transition examples.

The final transition baseline combines:

- a whole-verse transition classifier as the main decision-maker
- a localized transition model used for supporting span-level evidence
- a hard-case checkpoint promoted after explicit comparison against the previous baseline

Current transition baseline:

- overall accuracy: `0.901`
- `none = 0.896`
- `ikhfa = 0.921`
- `idgham = 0.857`

This module is now stable enough to be part of the official system baseline.

### 6.3 Burst Module

The burst module, focused mainly on `qalqalah`, was more stable than the duration and transition modules. It required less architectural intervention and reached a good baseline earlier in development.

Current burst baseline:

- overall accuracy: `0.874`
- `none = 0.914`
- `qalqalah = 0.814`

This module remains part of the current official baseline.

### 6.4 Content Module

The content module was the most difficult part of the project. The original full-verse content path was weak and showed strong deletion bias, especially on longer sequences. Error analysis showed that the model was frequently dropping characters and collapsing when asked to decode longer phrases.

To address this, the problem was reformulated:

- instead of full-verse content verification, the task was converted into chunked content verification
- shorter text-audio chunks were extracted and used for training and evaluation

This change produced a major improvement. After that, several further experiments were performed:

- hard-case oversampling for chunked content
- a larger chunked model on a stricter text-held-out split
- raw beam search decoding
- decoder tuning with blank penalties
- lexicon-constrained CTC decoding over the canonical chunk vocabulary

Among these, the best-performing solution was the lexicon-constrained decoder combined with the chunked content model.

Why this decoder matters:

- the content model outputs frame-level character probabilities
- the decoder is the stage that converts those probabilities into the final predicted text
- lexicon-constrained decoding improved performance by restricting the output to the canonical chunk vocabulary instead of allowing any arbitrary character sequence

Current content baseline:

- exact match: `0.738`
- character accuracy: `0.804`
- mean edit distance: `1.370`

On a stricter text-held-out split, the same content path achieved:

- exact match: `0.906`
- character accuracy: `0.932`
- mean edit distance: `0.468`

This is a substantial improvement over the earlier full-verse content baseline, which had:

- exact match: `0.016`
- character accuracy: `0.536`
- mean edit distance: `6.869`

## 7. Experimental Methodology

The project did not rely on isolated training runs or single reported scores. Instead, each meaningful change was evaluated against the current baseline before promotion. This included:

- module-level confusion analysis
- hard-case mining
- localized support experiments
- strict held-out validation
- side-by-side comparison of baseline versus candidate checkpoints
- integrated modular evaluation through the global suite

Equally important, several methods were explicitly rejected when they degraded performance. These include:

- chunked-content hard-case retraining
- a larger chunked content model on a stricter split
- raw beam-search decoding for chunked content

This means the current system baseline is not simply the most complex version. It is the version that consistently survived comparison gates.

## 8. Current Baseline System State

The current official baseline is:

- `duration`: approved learned fusion baseline
- `transition`: hard-case checkpoint with localized support
- `burst`: stable burst baseline
- `content`: chunked content model with lexicon-constrained decoding

The strongest current integrated system snapshot is:

- duration accuracy: `0.993` on the duration evaluation corpus
- transition accuracy: `0.901`
- burst accuracy: `0.874`
- content exact match: `0.738`
- content character accuracy: `0.804`
- content mean edit distance: `1.370`

For duration, the stricter generalization result that supports promotion remains the verse-held-out score of `0.973`.

## 9. Examples of Real Output Artifacts

In addition to aggregate metrics, the system now produces structured JSON outputs for both sample-level diagnosis and batch-level evaluation. These outputs were useful both for debugging and for validating the promotion of new baselines.

### 9.1 Example of structured error output

A typical evaluation artifact contains sample-level error entries with fields such as:

- `sample_id`
- `surah_name`
- `verse_key`
- `text`
- `source_module`
- `position`
- `char`
- `expected_rule`
- `predicted_rule`
- `confidence`
- `detail`

For example, one entry from the verse-held-out learned duration evaluation reports:

- `sample_id = retasy_train_000973`
- `surah_name = Al-Kauthar`
- `verse_key = verse_3`
- `text = "ان شانيك هو الابتر"`
- `source_module = duration`
- `position = 1`
- `char = "ن"`
- `expected_rule = ghunnah`
- `predicted_rule = madd`
- `confidence = 0.867`
- `detail = "expected ghunnah but got madd"`

This type of output is important because it shows that the system is not limited to global scores. It can also return interpretable, position-aware diagnostic information.

### 9.2 Example of batch evaluation output

The global batch evaluation JSON reports system-wide performance by module. For example, the current integrated suite reports:

- duration:
  - `samples = 973`
  - `accuracy = 0.993`
  - `ghunnah = 0.984`
  - `madd = 0.995`
- transition:
  - `samples = 690`
  - `accuracy = 0.901`
  - `ikhfa = 0.921`
  - `idgham = 0.857`
- burst:
  - `samples = 1597`
  - `accuracy = 0.874`
- content:
  - `samples = 389`
  - `decoder = lexicon`
  - `lexicon_size = 17`
  - `exact_match = 0.738`
  - `char_accuracy = 0.804`
  - `edit_distance = 1.370`

This means the evaluation stage is not limited to one final score. It provides module-level summaries, class-level summaries, and support statistics that help explain why a baseline was promoted.

### 9.3 Why these JSON outputs matter

These saved artifacts serve three purposes:

- they make the experiments reproducible
- they provide evidence for baseline promotion decisions
- they make the system explainable at both sample level and batch level

So the project now includes not only trained models, but also a structured analysis layer that supports technical interpretation.

## 10. Current Limitations

At this stage, the project is no longer blocked by missing architecture or broken integration. The main remaining limitation is broader generalization, especially on the content side.

The latest content improvement depends partly on a lexicon-constrained decoder using the canonical chunk vocabulary. This is a strong practical improvement for the current benchmark, but it also means that future work should evaluate how well the content module generalizes when the decoding space is less tightly constrained.

So the main remaining challenge is no longer basic system construction. It is the robustness and generalization of the content path beyond the current closed chunk vocabulary setup.

## 11. Conclusion

In conclusion, the development has progressed from a conceptual architecture to a working modular Tajweed assessment system with validated specialist modules, routing, aggregation, feedback generation, and systematic evaluation tools.

The major achievements so far are:

- successful implementation of the modular architecture
- successful use of MFCC-based features for temporal and local rule modeling
- successful use of wav2vec-style SSL representations for content verification
- stabilization and promotion of strong duration and transition baselines
- stable burst baseline
- major rescue and improvement of the content module through chunking and constrained decoding
- consistent experimental methodology with explicit promotion and rejection of candidate improvements

The project has therefore moved beyond the prototype stage. It now has a coherent and defensible baseline system, with the next phase focused mainly on improving content generalization rather than building the architecture itself.
