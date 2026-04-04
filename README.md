# Tajweed Modular Assessment

Minimal but runnable scaffold for a modular Tajweed assessment system.

What is implemented now:
- a working **MFCC + BiLSTM** duration-rule baseline
- dual heads: **phoneme CTC** + **rule classification**
- a simple **content-first aggregation** layer
- lightweight transition and burst specialist modules
- toy datasets so the repo runs immediately
- tests for dataset, alignment, aggregation, and model forwards

What is still a scaffold:
- real Arabic wav2vec content verification
- true CTC forced alignment / Viterbi
- real Quran recitation manifests and JSON labels
- production-grade feature routing

Quick start:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_duration.py
python scripts/run_inference.py
pytest -q
```
