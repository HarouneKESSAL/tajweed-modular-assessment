# Content Multitask Regression Analysis

## Compared checkpoints

Baseline:

- content_chunked_module_hd96_reciter.pt

Candidates:

- content_multitask_word_chunk_balanced_only.pt
- content_multitask_word_chunk_finetune_safe_v1.pt

## Balanced-only result

The balanced-only checkpoint slightly improves character-level metrics but worsens exact match.

Regression script result on 100 validation chunks:

- baseline_exact: 0.470
- candidate_exact: 0.460
- baseline_char: 0.679
- candidate_char: 0.684
- baseline_edit: 2.170
- candidate_edit: 2.060
- broken: 3
- fixed: 2
- both_wrong_improved: 14
- both_wrong_worsened: 10

Suite result on content-limit 100:

- baseline exact_match: 0.780
- baseline char_accuracy: 0.893
- baseline edit_distance: 0.620
- baseline critical content errors: 22

- balanced_only exact_match: 0.760
- balanced_only char_accuracy: 0.894
- balanced_only edit_distance: 0.560
- balanced_only critical content errors: 24

Interpretation:

The model improves some hard wrong predictions, but it introduces slightly more exact-match regressions. Since content errors are weighted as critical, this checkpoint is not promoted.

## Safe-v1 result

Safe-v1 is worse and should not be continued.

Regression script result:

- baseline_exact: 0.470
- candidate_exact: 0.390
- baseline_char: 0.679
- candidate_char: 0.639
- broken: 9
- fixed: 1
- both_wrong_worsened: 20
- both_wrong_improved: 15

## Observed error pattern

Balanced-only fixed examples include:

- وايانستعين -> واياكنستعين
- الناس -> الخناس

Balanced-only broken examples include:

- واياكنستعين -> وايالنستعين
- الرحمن -> ارحمن
- منالجنة -> منالجة

The model learns useful character corrections but is not stable enough for exact Quranic content recognition.

## Decision

Do not promote either multitask checkpoint yet.

Keep the original checkpoint promoted:

- content_chunked_module_hd96_reciter.pt

## Next direction

Before more training, align the regression comparison script with the suite decoder. Then try a lower-risk training strategy:

- initialize from baseline checkpoint
- use very small learning rate
- use only a small balanced auxiliary word phase
- select checkpoints by exact match and critical content errors, not char accuracy
