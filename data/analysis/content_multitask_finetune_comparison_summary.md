# Multitask Word + Chunk Content Fine-Tuning Experiments

## Baseline on content-limit 100

- checkpoint: content_chunked_module_hd96_reciter.pt
- exact_match: 0.780
- char_accuracy: 0.893
- edit_distance: 0.620
- critical content errors: 22
- content weighted penalty: 220

## Balanced-only fine-tune

- checkpoint: content_multitask_word_chunk_balanced_only.pt
- exact_match: 0.760
- char_accuracy: 0.894
- edit_distance: 0.560
- critical content errors: 24
- content weighted penalty: 240

Interpretation: improves char accuracy and edit distance, but worsens exact match and critical weighted errors. Not promoted.

## Safe v1 fine-tune

- checkpoint: content_multitask_word_chunk_finetune_safe_v1.pt
- exact_match: 0.720
- char_accuracy: 0.894
- edit_distance: 0.560
- critical content errors: 28
- content weighted penalty: 280

Interpretation: not promoted.

## Conclusion

Checkpoint-initialized multitask fine-tuning is technically valid and promising, but current curricula do not beat the baseline under weighted Tajweed scoring.

The original content checkpoint remains promoted.
