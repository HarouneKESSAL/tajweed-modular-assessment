# Content Checkpoint Regression Analysis

## Checkpoints

- Baseline: `C:\Users\anis\Desktop\tajweed-modular-assessment\checkpoints\content_chunked_module_hd96_reciter.pt`
- Candidate: `C:\Users\anis\Desktop\tajweed-modular-assessment\checkpoints\content_multitask_word_chunk_finetune_safe_v1.pt`

## Summary

- Samples: 100
- Baseline exact: 0.470
- Candidate exact: 0.390
- Baseline char accuracy: 0.679
- Candidate char accuracy: 0.639
- Baseline mean edit distance: 2.170
- Candidate mean edit distance: 2.350

## Regression Categories

- broken: 9
- fixed: 1
- both_wrong_worsened: 20
- both_wrong_improved: 15
- both_wrong_same: 17
- both_correct: 38

## Fixed examples

### retasy_train_005696_chunk_01

- Gold: `واياكنستعين`
- Baseline: `اياكنستعين`
- Candidate: `واياكنستعين`
- Base edit: 1
- Candidate edit: 0
- Surah: Al-Faatihah
- Verse: verse_5


## Broken examples

### retasy_train_005591_chunk_00

- Gold: `الرحمن`
- Baseline: `الرحمن`
- Candidate: `امن`
- Base edit: 0
- Candidate edit: 3
- Surah: Al-Faatihah
- Verse: verse_3

### retasy_train_005629_chunk_00

- Gold: `منالجنة`
- Baseline: `منالجنة`
- Candidate: `مناة`
- Base edit: 0
- Candidate edit: 3
- Surah: An-Nas
- Verse: verse_6

### retasy_train_005371_chunk_01

- Gold: `واياكنستعين`
- Baseline: `واياكنستعين`
- Candidate: `واياستعين`
- Base edit: 0
- Candidate edit: 2
- Surah: Al-Faatihah
- Verse: verse_5

### retasy_train_005429_chunk_00

- Gold: `منشر`
- Baseline: `منشر`
- Candidate: `نر`
- Base edit: 0
- Candidate edit: 2
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005429_chunk_01

- Gold: `الوسواس`
- Baseline: `الوسواس`
- Candidate: `الواس`
- Base edit: 0
- Candidate edit: 2
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005472_chunk_00

- Gold: `انشانيك`
- Baseline: `انشانيك`
- Candidate: `انيشايك`
- Base edit: 0
- Candidate edit: 2
- Surah: Al-Kauthar
- Verse: verse_3

### retasy_train_005542_chunk_00

- Gold: `اياكنعبد`
- Baseline: `اياكنعبد`
- Candidate: `اياعبد`
- Base edit: 0
- Candidate edit: 2
- Surah: Al-Faatihah
- Verse: verse_5

### retasy_train_005472_chunk_01

- Gold: `هوالابتر`
- Baseline: `هوالابتر`
- Candidate: `هوالبتر`
- Base edit: 0
- Candidate edit: 1
- Surah: Al-Kauthar
- Verse: verse_3

### retasy_train_005610_chunk_00

- Gold: `منشر`
- Baseline: `منشر`
- Candidate: `منر`
- Base edit: 0
- Candidate edit: 1
- Surah: An-Nas
- Verse: verse_4


## Improved but still wrong

### retasy_train_005644_chunk_00

- Gold: `منشر`
- Baseline: `منكالجنة`
- Candidate: `منشجر`
- Base edit: 6
- Candidate edit: 1
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

### retasy_train_005677_chunk_00

- Gold: `منشر`
- Baseline: `منالجناس`
- Candidate: `مة`
- Base edit: 6
- Candidate edit: 3
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005547_chunk_01

- Gold: `الرحيم`
- Baseline: `واياكنعبد`
- Candidate: `عاعبد`
- Base edit: 8
- Candidate edit: 6
- Surah: Al-Faatihah
- Verse: verse_3

### retasy_train_005610_chunk_01

- Gold: `الوسواس`
- Baseline: `واياكنسعن`
- Candidate: `واياكن`
- Base edit: 8
- Candidate edit: 6
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005626_chunk_01

- Gold: `عابدما`
- Baseline: `ارحعبمن`
- Candidate: `اعمد`
- Base edit: 6
- Candidate edit: 4
- Surah: Al-Kafiroon
- Verse: verse_4

### retasy_train_005640_chunk_00

- Gold: `الرحمن`
- Baseline: `انيالجنعيد`
- Candidate: `انيالبد`
- Base edit: 8
- Candidate edit: 6
- Surah: Al-Faatihah
- Verse: verse_3

### retasy_train_005361_chunk_00

- Gold: `لااعبد`
- Baseline: `واياكنباس`
- Candidate: `واياناس`
- Base edit: 6
- Candidate edit: 5
- Surah: Al-Kafiroon
- Verse: verse_2

### retasy_train_005377_chunk_00

- Gold: `منشر`
- Baseline: `والوساس`
- Candidate: `والوسة`
- Base edit: 7
- Candidate edit: 6
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005377_chunk_01

- Gold: `الوسواس`
- Baseline: `واياكنوعبد`
- Candidate: `واياوعبد`
- Base edit: 8
- Candidate edit: 7
- Surah: An-Nas
- Verse: verse_4


## Worsened and still wrong

### retasy_train_005699_chunk_02

- Gold: `الخناس`
- Baseline: `والنا`
- Candidate: `و`
- Base edit: 3
- Candidate edit: 6
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005367_chunk_01

- Gold: `هوالابتر`
- Baseline: `مالابون`
- Candidate: `الون`
- Base edit: 4
- Candidate edit: 6
- Surah: Al-Kauthar
- Verse: verse_3

### retasy_train_005424_chunk_01

- Gold: `واياكنستعين`
- Baseline: `وايانستعين`
- Candidate: `واياتعين`
- Base edit: 1
- Candidate edit: 3
- Surah: Al-Faatihah
- Verse: verse_5

### retasy_train_005429_chunk_02

- Gold: `الخناس`
- Baseline: `الناس`
- Candidate: `والاس`
- Base edit: 1
- Candidate edit: 3
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005503_chunk_00

- Gold: `ملكالناس`
- Baseline: `مالرحان`
- Candidate: `ولة`
- Base edit: 5
- Candidate edit: 7
- Surah: An-Nas
- Verse: verse_2

### retasy_train_005610_chunk_02

- Gold: `الخناس`
- Baseline: `الناس`
- Candidate: `والاس`
- Base edit: 1
- Candidate edit: 3
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005648_chunk_02

- Gold: `الخناس`
- Baseline: `الناس`
- Candidate: `والاس`
- Base edit: 1
- Candidate edit: 3
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005361_chunk_01

- Gold: `ماتعبدون`
- Baseline: `الكاناس`
- Candidate: ``
- Base edit: 7
- Candidate edit: 8
- Surah: Al-Kafiroon
- Verse: verse_2

### retasy_train_005377_chunk_02

- Gold: `الخناس`
- Baseline: `الناس`
- Candidate: `الاس`
- Base edit: 1
- Candidate edit: 2
- Surah: An-Nas
- Verse: verse_4

### retasy_train_005416_chunk_00

- Gold: `ملكالناس`
- Baseline: `والناس`
- Candidate: `والاس`
- Base edit: 3
- Candidate edit: 4
- Surah: An-Nas
- Verse: verse_2

