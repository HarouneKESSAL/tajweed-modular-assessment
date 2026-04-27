# Presentation Report: What We Built In The Code

This report explains, in plain language, what was implemented from the original conceptual framework and how the system evolved from the initial design into the current working baseline.

## 1. Starting Point

The work started from the architecture described in [Conceptual_Framework.pdf](C:/Users/anis/Desktop/tajweed-modular-assessment/Conceptual_Framework.pdf). That document did not describe a single monolithic model. Instead, it proposed a modular Tajweed assessment system with five main parts: feature extraction, content verification, rule analysis modules, diagnosis aggregation, and feedback generation. It also explicitly included a routing step, where the system uses the content alignment to decide which audio segments should be checked by which Tajweed rule module.

So from the beginning, the implementation goal was not simply “train a classifier.” The real goal was to translate the conceptual design into a working pipeline that could process recitation audio, verify the content, route relevant segments to specialist rule modules, combine the outputs, and produce structured feedback.

## 2. First Implementation Phase: Building the Modular Pipeline

The first major step in the code was to make the architecture real. That meant implementing the modular inference pipeline itself. We built code that accepts recitation audio, extracts features, runs content verification, performs routing, invokes specialist rule modules, aggregates all judgments, and formats feedback. This was the point where the project changed from a theoretical design into an executable system.

A key design choice in the implementation was to preserve the PDF’s separation of concerns. Instead of trying to force one model to handle all Tajweed phenomena, the code was organized into specialist paths:

- a duration path for rules such as `madd` and `ghunnah`
- a transition path for rules such as `ikhfa` and `idgham`
- a burst path for rules such as `qalqalah`
- a content-verification path to detect transcription-level errors

This matters because these error types are acoustically different. Duration rules depend on how long a sound is held, transition rules depend on how sounds merge or conceal into one another, burst rules depend on brief transient articulatory events, and content errors depend on what was actually recited relative to the canonical text.

## 3. Making the System Behave Correctly

After the basic modular pipeline existed, a large amount of work went into making it behave correctly. The system initially had several integration problems. For example, some early rule-only runs were polluted by placeholder content behavior, which led to meaningless deletion floods in the diagnosis report. We fixed that so rule-only samples are handled as rule-only samples rather than being forced through a fake content alignment.

We also improved the reporting layer itself. The raw outputs were not useful enough for debugging or presentation, so the code was extended to include:

- character-aware error reporting
- confidence values on rule judgments
- matched findings for positions that were correctly recognized
- hybrid evidence summaries when multiple models contribute to a decision

That work was important because without it, the pipeline could run but it would still be difficult to inspect, explain, or improve.

## 4. Duration Module Development

The duration module was one of the first specialist modules to become stable. It focuses mainly on `madd` and `ghunnah`, which are both temporal rules. The baseline duration model performed very well on `madd`, but it was significantly weaker on `ghunnah`, especially when `ghunnah` was confused with `madd`.

Several approaches were tested to improve this:

- class weighting
- hard-case mining and retraining
- localized duration support
- conservative rule-based fusion
- learned fusion calibration

The class-weighting and hard-case retraining experiments improved `ghunnah` in some cases but damaged `madd` too much. That meant the overall duration baseline did not actually improve enough to justify promotion. To solve this more cleanly, a localized duration model was introduced as supporting evidence rather than as a replacement. This made it possible to compare the main sequence-level duration judgment against local segment-level evidence.

After that, a learned fusion calibrator was built. The calibrator decides when local evidence should override or support the main duration prediction. This was treated as experimental at first, because one early split produced suspiciously perfect numbers. Instead of trusting that, a stricter verse-held-out evaluation was built. Only after the learned fusion beat the conservative duration baseline on that stricter split was it promoted as the approved duration-fusion path.

So the duration system did not become strong by chance. It became strong because every candidate improvement was tested, and only the one that survived a stricter approval gate was promoted.

## 5. Transition Module Development

The transition module covers rules such as `ikhfa` and `idgham`. The first issue here was instability: repeated evaluations could produce inconsistent results because a dummy SSL projection was not deterministic. That was fixed first. Without deterministic behavior, none of the later evaluation results would have been reliable.

Once the transition path became stable, confusion analysis showed that the main errors involved false positives on `ikhfa` and misses on `idgham`. A localized transition model was built to provide span-level evidence, but it was not strong enough to replace the whole-verse transition model as the main decision-maker. So the system kept the whole-verse classifier as the primary transition decision and used the localized model as supporting evidence.

The biggest improvement came from hard-case mining and retraining. The transition mistakes were mined, weighted, and used to train a focused alternative checkpoint. That significantly improved the transition results, and the hard-case checkpoint became the promoted default. The final transition path is therefore hybrid in a practical sense: the whole-verse transition classifier makes the main rule decision, while the localized transition model provides extra evidence and span information for interpretation.

## 6. Burst Module Development

The burst module, focused on `qalqalah`, was comparatively more stable. It did not require the same amount of architectural experimentation as duration or transition. It remains part of the final baseline as the current burst specialist. This module contributed to the modular structure by showing that some Tajweed categories could be modeled cleanly once the routing and evaluation infrastructure were in place.

## 7. Content Verification Development

The content path was the weakest major subsystem for a long time. Initially, the full-verse content model performed badly. Error analysis showed that it was strongly deletion-biased and tended to collapse on longer phrases. In practical terms, this meant the model often lost parts of the verse or failed to maintain a stable output across long targets.

Instead of continuing to optimize the full-verse content path, the implementation changed the problem structure. A chunked content pipeline was introduced. This breaks the content task into shorter audio-text chunks, which makes the recognition problem easier and reduces long-sequence collapse. That change produced a large improvement over the original full-verse baseline.

After the chunked pipeline existed, decoder tuning was explored. The most effective improvement was to apply a CTC blank penalty during greedy decoding. That produced a measurable gain in exact match, character accuracy, and edit distance, so it became the promoted chunked decoder baseline.

Other content experiments were also tried:

- hard-case oversampling for chunked training
- a larger chunked model on a stricter text-held-out split
- raw beam-search decoding

All three were worse than the tuned greedy chunked baseline, so they were rejected. This is an important point for the presentation: the project did not improve by blindly increasing complexity. Several more advanced-looking options were tested and explicitly rejected because they made the baseline worse.

## 8. Aggregation and Feedback

Another major part of the work was the aggregation layer. The specialist modules do not operate in isolation. A rule judgment may be misleading if the recitation already contains a content error on the same region. The aggregation layer resolves these interactions and builds a structured diagnosis report.

On top of that, the feedback generator translates the structured report into a readable explanation. This includes rule names, positions, characters, confidence values, and localized supporting evidence where relevant. That made the system not only analyzable during development, but also explainable to a user or learner.

## 9. Final State of the Codebase

At the current stage, the codebase is no longer a prototype made of isolated experiments. It is a modular assessment system with:

- a working end-to-end pipeline
- routing logic based on content alignment
- specialist rule modules
- aggregation logic
- human-readable feedback
- evaluation scripts
- confusion and failure analyzers
- promoted baselines that were selected through explicit comparison gates

The strongest currently promoted system state is:

- duration with approved learned fusion
- transition with the hard-case checkpoint and localized support
- burst with the current burst baseline
- content with the chunked model and lexicon-constrained decoder

## 10. Main Remaining Limitation

The main remaining weakness in the system is no longer a clearly broken module. Duration, transition, and burst have all reached strong modular baselines, and content improved substantially after chunking and then again after lexicon-constrained decoding. The remaining limitation is that content performance now depends on a closed canonical chunk vocabulary, which is strong for the current benchmark but still a narrower setting than open-ended recitation recognition.

That means the project is no longer blocked by architectural uncertainty. The modular architecture from the PDF has already been implemented and validated. The remaining work is now concentrated on improving the quality of the content-verification path.

## 11. Short Conclusion

In summary, the work done in code was not just “train some models.” The real achievement was to implement the PDF’s modular architecture faithfully, make the modules cooperate correctly, evaluate them systematically, and promote only those changes that genuinely improved the baseline. The result is a working modular Tajweed assessment system with promoted baselines for duration, transition, burst, and content, plus a clear understanding of the remaining generalization limits.
