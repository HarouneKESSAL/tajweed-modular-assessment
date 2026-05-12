# Content Module Promotion: Tiny Auxiliary Multitask v2

## Decision

Promote `content_multitask_word_chunk_tiny_aux_v2.pt` as the new content module candidate.

The previous official baseline remains preserved as:

- `content_chunked_module_hd96_reciter.pt`

## Baseline full validation

- checkpoint: `content_chunked_module_hd96_reciter.pt`
- exact_match: 0.700
- char_accuracy: 0.892
- edit_distance: 0.731
- critical content errors: 125
- content weighted penalty: 1250
- estimated weighted average: 98.570

## Tiny auxiliary v2 full validation

- checkpoint: `content_multitask_word_chunk_tiny_aux_v2.pt`
- exact_match: 0.707
- char_accuracy: 0.893
- edit_distance: 0.724
- critical content errors: 122
- content weighted penalty: 1220
- estimated weighted average: 98.591

## Interpretation

The improvement is small but consistent across all important metrics.

The v2 model improves:

- exact content recognition
- character accuracy
- edit distance
- critical content error count
- weighted content penalty
- overall severity-aware score

## Training recipe

The successful recipe was conservative:

- initialize from `content_chunked_module_hd96_reciter.pt`
- reuse checkpoint character vocabulary
- use Wav2Vec/SSL + CTC architecture
- train only one tiny auxiliary multitask stage
- use 50 word samples + 200 chunk samples
- learning rate: 0.00001
- epochs: 1

## Important lesson

Larger or more aggressive multitask fine-tuning hurt exact content reliability.

Successful direction:

- small auxiliary word exposure
- mostly chunk-preserving training
- low learning rate
- select by exact match and weighted critical content errors, not only char accuracy

## Promotion status

Promoted as content candidate v2.

Do not delete the original baseline checkpoint.
