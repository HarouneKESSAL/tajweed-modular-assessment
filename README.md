# Tajweed Modular Assessment

Repository scaffold for a modular automatic Tajweed assessment system.

## Design goals
- Separate content verification from rule analysis.
- Keep rule analysis modular by acoustic category.
- Start with a BiLSTM baseline + CTC alignment + rule classification.
- Preserve an upgrade path to wav2vec/HuBERT and additional specialist modules.
- Produce a structured JSON diagnosis before generating learner-facing feedback.

## Current intended implementation order
1. Data manifests + Coloured Qur'an JSON parsing
2. MFCC pipeline + BiLSTM duration baseline
3. Shared encoder with dual heads (CTC + rule classification)
4. Content verifier and alignment utilities
5. Diagnosis aggregation
6. Transition and burst specialists
7. Feedback generation

## Quick tree
See `docs/repo_tree.txt`.
