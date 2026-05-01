# Final Demo Report

Generated at: `2026-05-01T17:36:48`

## Official Baseline

| Module | Status | Main Result | Notes |
| --- | --- | --- | --- |
| Duration | promoted | verse-held-out acc `0.973`; suite acc `0.993` | Use verse-held-out score for generalization. |
| Transition | promoted | acc `0.901` | Hard-case checkpoint promoted. |
| Burst | baseline | acc `0.874` | Qalqalah/burst baseline. |
| Content | official_chunked_open_baseline | exact `0.700`, char acc `0.892` | Chunked open baseline; learner-level content not promoted. |

## Experimental Findings

- Distilled content bridge tied the baseline exact match at `0.700`, but new aligned chunk char accuracy stayed low at `0.224`.
- Learner-level content recognition is not ready: expanded recognizer validation exact match was `0.019` and char accuracy was `0.276`.

## What To Say

- We implemented the PDF architecture as a modular Tajweed assessment pipeline.
- Routing chooses the correct specialist module: duration, transition, burst, or content.
- MFCC features support rule modules; wav2vec-style SSL features support content recognition.
- We promoted only changes that passed explicit comparison gates.
- Duration, transition, and burst are usable specialist baselines.
- Content improved from weak full-verse recognition to a stronger chunked open baseline.
- Learner-level open content recognition remains future work because expanded recognizers do not yet generalize enough.

## Demo Commands

```powershell
.\.venv\Scripts\python scripts\system\generate_demo_report.py
```
```powershell
.\.venv\Scripts\python scripts\system\run_inference.py --manifest data\manifests\retasy_transition_subset.jsonl --sample-index 1 --show-matches
```
```powershell
.\.venv\Scripts\python scripts\system\evaluate_modular_suite.py --chunked-content-checkpoint content_chunked_module_hd96_reciter.pt --content-split-mode text --content-decoder-config checkpoints\content_chunked_decoder_open_hd96.json --output-json data\analysis\demo_modular_suite_open_hd96.json
```
