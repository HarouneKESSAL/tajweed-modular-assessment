# Content Checkpoint Regression Analysis

## Checkpoints

- Baseline: `C:\Users\anis\Desktop\tajweed-modular-assessment\checkpoints\content_chunked_module_hd96_reciter.pt`
- Candidate: `C:\Users\anis\Desktop\tajweed-modular-assessment\checkpoints\content_multitask_word_chunk_balanced_only.pt`

## Summary

- Samples: 100
- Baseline exact: 0.470
- Candidate exact: 0.460
- Baseline char accuracy: 0.679
- Candidate char accuracy: 0.684
- Baseline mean edit distance: 2.170
- Candidate mean edit distance: 2.060

## Regression Categories

- broken: 3
- fixed: 2
- both_wrong_worsened: 10
- both_wrong_improved: 14
- both_wrong_same: 27
- both_correct: 44

## Fixed examples

### retasy_train_005424_chunk_01

- Gold: `واياكنستعين`
- Baseline: `وايانستعين`
- Candidate: `واياكنستعين`
- Base edit: 1
- Candidate edit: 0
- Surah: Al-Faatihah
- Verse: verse_5

### retasy_train_005610_chunk_02

- Gold: `الخناس`
- Baseline: `الناس`
- Candidate: `الخناس`
- Base edit: 1
- Candidate edit: 0
- Surah: An-Nas
- Verse: verse_4


## Broken examples

### retasy_train_005371_chunk_01

- Gold: `واياكنستعين`
- Baseline: `واياكنستعين`
- Candidate: `وايالنستعين`
- Base edit: 0
- Candidate edit: 1
- Surah: Al-Faatihah
- Verse: verse_5

### retasy_train_005591_chunk_00

- Gold: `الرحمن`
- Baseline: `الرحمن`
- Candidate: `ارحمن`
- Base edit: 0
- Candidate edit: 1
- Surah: Al-Faatihah
- Verse: verse_3

### retasy_train_005629_chunk_00

- Gold: `منالجنة`
- Baseline: `منالجنة`
- Candidate: `منالجة`
- Base edit: 0
- Candidate edit: 1
- Surah: An-Nas
- Verse: verse_6


## Improved but still wrong

### retasy_train_005677_chunk_00

- Gold: `منشر`
- Baseline: `منالجناس`
- Candidate: `منلة`
- Base edit: 6
- Candidate edit: 2
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005430_chunk_00

- Gold: `ولاانا`
- Baseline: `لالنعبدما`
- Candidate: `لاما`
- Base edit: 6
- Candidate edit: 3
- Surah: Al-Kafiroon
- Verse: verse_4

### retasy_train_005547_chunk_01

- Gold: `الرحيم`
- Baseline: `واياكنعبد`
- Candidate: `عاعبد`
- Base edit: 8
- Candidate edit: 6
- Surah: Al-Faatihah
- Verse: verse_3

### retasy_train_005626_chunk_01

- Gold: `عابدما`
- Baseline: `ارحعبمن`
- Candidate: `اعم`
- Base edit: 6
- Candidate edit: 4
- Surah: Al-Kafiroon
- Verse: verse_4

### retasy_train_005644_chunk_00

- Gold: `منشر`
- Baseline: `منكالجنة`
- Candidate: `منشالجنر`
- Base edit: 6
- Candidate edit: 4
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005699_chunk_01

- Gold: `الوسواس`
- Baseline: `الحسعيان`
- Candidate: `الساس`
- Base edit: 4
- Candidate edit: 2
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005377_chunk_00

- Gold: `منشر`
- Baseline: `والوساس`
- Candidate: `والوسس`
- Base edit: 7
- Candidate edit: 6
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005430_chunk_01

- Gold: `عابدما`
- Baseline: `اكعبدود`
- Candidate: `اعبدون`
- Base edit: 5
- Candidate edit: 4
- Surah: Al-Kafiroon
- Verse: verse_4

### retasy_train_005430_chunk_02

- Gold: `عبدتم`
- Baseline: `الرحيم`
- Candidate: `الرحم`
- Base edit: 5
- Candidate edit: 4
- Surah: Al-Kafiroon
- Verse: verse_4

### retasy_train_005444_chunk_00

- Gold: `منشر`
- Baseline: `الرحم`
- Candidate: `الرح`
- Base edit: 5
- Candidate edit: 4
- Surah: An-Nas
- Verse: verse_4


## Worsened and still wrong

### retasy_train_005429_chunk_02

- Gold: `الخناس`
- Baseline: `الناس`
- Candidate: `والاس`
- Base edit: 1
- Candidate edit: 3
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005367_chunk_01

- Gold: `هوالابتر`
- Baseline: `مالابون`
- Candidate: `مالاون`
- Base edit: 4
- Candidate edit: 5
- Surah: Al-Kauthar
- Verse: verse_3

### retasy_train_005416_chunk_00

- Gold: `ملكالناس`
- Baseline: `والناس`
- Candidate: `والاس`
- Base edit: 3
- Candidate edit: 4
- Surah: An-Nas
- Verse: verse_2

### retasy_train_005556_chunk_00

- Gold: `الرحمن`
- Baseline: `لرحعبان`
- Candidate: `لرحعباس`
- Base edit: 4
- Candidate edit: 5
- Surah: Al-Faatihah
- Verse: verse_3

### retasy_train_005563_chunk_00

- Gold: `ملكالناس`
- Baseline: `الرحعباس`
- Candidate: `الرح`
- Base edit: 5
- Candidate edit: 6
- Surah: An-Nas
- Verse: verse_2

### retasy_train_005581_chunk_01

- Gold: `واياكنستعين`
- Baseline: `وايلنستعين`
- Candidate: `وايلستعين`
- Base edit: 2
- Candidate edit: 3
- Surah: Al-Faatihah
- Verse: verse_5

### retasy_train_005644_chunk_02

- Gold: `الخناس`
- Baseline: `والناس`
- Candidate: `واناس`
- Base edit: 2
- Candidate edit: 3
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005648_chunk_01

- Gold: `الوسواس`
- Baseline: `الوسوعيس`
- Candidate: `الووعيس`
- Base edit: 2
- Candidate edit: 3
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005648_chunk_02

- Gold: `الخناس`
- Baseline: `الناس`
- Candidate: `الواس`
- Base edit: 1
- Candidate edit: 2
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005699_chunk_02

- Gold: `الخناس`
- Baseline: `والنا`
- Candidate: `ولن`
- Base edit: 3
- Candidate edit: 4
- Surah: An-Nas
- Verse: verse_4

