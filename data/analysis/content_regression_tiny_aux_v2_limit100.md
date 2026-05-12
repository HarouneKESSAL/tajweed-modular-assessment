# Content Checkpoint Regression Analysis

## Checkpoints

- Baseline: `C:\Users\anis\Desktop\tajweed-modular-assessment\checkpoints\content_chunked_module_hd96_reciter.pt`
- Candidate: `C:\Users\anis\Desktop\tajweed-modular-assessment\checkpoints\content_multitask_word_chunk_tiny_aux_v2.pt`

## Summary

- Samples: 100
- Baseline exact: 0.470
- Candidate exact: 0.470
- Baseline char accuracy: 0.679
- Candidate char accuracy: 0.678
- Baseline mean edit distance: 2.170
- Candidate mean edit distance: 2.170

## Regression Categories

- both_wrong_worsened: 3
- both_wrong_improved: 3
- both_wrong_same: 47
- both_correct: 47

## Fixed examples

- None

## Broken examples

- None

## Improved but still wrong

### retasy_train_005430_chunk_00

- Gold: `ولاانا`
- Baseline: `لالنعبدما`
- Candidate: `لاانعبدما`
- Base edit: 6
- Candidate edit: 5
- Surah: Al-Kafiroon
- Verse: verse_4

### retasy_train_005570_chunk_00

- Gold: `الرحمن`
- Baseline: `انلي`
- Candidate: `انليم`
- Base edit: 5
- Candidate edit: 4
- Surah: Al-Faatihah
- Verse: verse_3

### retasy_train_005644_chunk_00

- Gold: `منشر`
- Baseline: `منكالجنة`
- Candidate: `منكالجنر`
- Base edit: 6
- Candidate edit: 5
- Surah: An-Nas
- Verse: verse_4


## Worsened and still wrong

### retasy_train_005367_chunk_01

- Gold: `هوالابتر`
- Baseline: `مالابون`
- Candidate: `مالاون`
- Base edit: 4
- Candidate edit: 5
- Surah: Al-Kauthar
- Verse: verse_3

### retasy_train_005556_chunk_00

- Gold: `الرحمن`
- Baseline: `لرحعبان`
- Candidate: `لرحعياس`
- Base edit: 4
- Candidate edit: 5
- Surah: Al-Faatihah
- Verse: verse_3

### retasy_train_005581_chunk_01

- Gold: `واياكنستعين`
- Baseline: `وايلنستعين`
- Candidate: `والنستعين`
- Base edit: 2
- Candidate edit: 3
- Surah: Al-Faatihah
- Verse: verse_5

