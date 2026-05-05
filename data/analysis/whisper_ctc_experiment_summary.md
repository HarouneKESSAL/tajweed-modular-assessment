# Whisper Encoder CTC Content Experiment

## Goal

Test whether Whisper encoder features plus a Quran-specific CTC head can improve the content module.

## Baseline to beat

Official chunked content baseline:

- exact_match: 0.700
- char_accuracy: 0.892

## Experiment 1: Whisper tiny + linear CTC head

- max rows: 200
- train samples: 172
- val samples: 28
- freeze_encoder: true
- full validation char_accuracy: about 0.223
- exact_match: 0.000

## Experiment 2: Whisper tiny + BiLSTM CTC head

- max rows: 200
- train samples: 172
- val samples: 28
- freeze_encoder: true
- same-split char_accuracy: 0.235
- full validation char_accuracy: 0.225
- exact_match: 0.000

## Interpretation

The Whisper-CTC path is technically functional, but frozen Whisper tiny features are not strong enough for the Quran chunk content task at this scale.

The experiment is not promoted.

## Next direction

Use the existing content architecture and improve it through multitask training with Quran-MD word clips plus Retasy content chunks.

Whisper may be revisited later with:

- Whisper-small encoder
- GPU training
- partial encoder unfreezing
- more data
