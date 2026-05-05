# Whisper Small Cropped Content Experiment

## Goal

Test whether OpenAI Whisper can directly replace or improve the current chunked content module.

## Setup

- Model: openai/whisper-small
- Device: CPU
- Samples: 20
- Audio: cropped content chunks using start_sec/end_sec
- Max new tokens: 24

## Result

- exact_match: 0.000
- normalized_exact_match: 0.000
- char_accuracy: 0.268
- normalized_char_accuracy: 0.288
- mean_edit_distance: 10.250
- normalized_mean_edit_distance: 10.100

## Interpretation

Chunk cropping is now confirmed to work because audio_to_whisper_sec matches chunk_sec.

However, Whisper zero-shot is still far below the current official content baseline:

- current baseline exact_match: 0.700
- current baseline char_accuracy: 0.892

Therefore, Whisper decoder should not be promoted as a replacement.

## Next direction

Use Whisper only as a feature encoder later, with a Quran-specific CTC head, or continue with multitask word+chunk training.
