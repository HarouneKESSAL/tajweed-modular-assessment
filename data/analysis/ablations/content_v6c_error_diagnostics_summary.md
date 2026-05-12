# Chunk content error diagnostics

## Overall

| metric | value |
|---|---:|
| samples | 417 |
| exact_rate | 0.707 |
| avg_char_accuracy | 0.893 |
| empty_pred_rate | 0.000 |
| avg_gold_len | 6.470 |
| avg_pred_len | 6.295 |
| num_errors | 122 |
| num_empty_predictions | 0 |
| num_too_short | 24 |
| num_too_long | 13 |

## By gold length

| bucket | samples | exact_rate | char_acc | empty_pred_rate | avg_gold_len | avg_pred_len |
|---|---:|---:|---:|---:|---:|---:|
| 004_006 | 48 | 0.000 | 0.594 | 0.000 | 5.50 | 5.62 |
| 007_010 | 74 | 0.000 | 0.660 | 0.000 | 7.46 | 6.39 |

## Top repeated gold failures

| gold | count | avg_char_acc | common predictions |
|---|---:|---:|---|
| `الوسواس` | 40 | 0.714 | `الوساس` x16; `الواس` x8; `الوواس` x5; `والخناس` x1; `السواس` x1 |
| `لااعبد` | 30 | 0.711 | `لاعبد` x21; `اعبد` x1; `اليد` x1; `لياعبد` x1; `لاعاد` x1 |
| `ملكالناس` | 18 | 0.569 | `ملناس` x3; `والناس` x3; `الكالناس` x2; `ملكاناس` x1; `النمن` x1 |
| `ماتعبدون` | 16 | 0.625 | `ماعبدون` x6; `ماتعبدتن` x2; `والنياس` x1; `التين` x1; `ماتتعبدون` x1 |
| `منشر` | 12 | 0.292 | `منشار` x2; `منش` x1; `منشرنس` x1; `منشرر` x1; `والوساس` x1 |
| `والناس` | 6 | 0.611 | `الناس` x2; `والخناس` x1; `والنعاس` x1; `الرنبيمن` x1; `عالرنعم` x1 |

## Character-level error clues

- top deleted gold chars: `[('ا', 33), ('و', 31), ('س', 20), ('ل', 15), ('ك', 7), ('ت', 7), ('م', 6), ('ر', 4), ('ش', 3), ('ع', 3), ('ن', 1), ('ب', 1)]`
- top inserted pred chars: `[('ا', 30), ('ن', 22), ('و', 15), ('ي', 15), ('ع', 12), ('ر', 10), ('س', 9), ('ل', 9), ('ك', 9), ('ح', 8), ('م', 7), ('ب', 4), ('ت', 4), ('ة', 3), ('د', 3), ('ج', 2), ('خ', 2)]`
- top replaced gold chars: `[('و', 20), ('م', 13), ('س', 12), ('ل', 9), ('ا', 8), ('ع', 8), ('د', 8), ('ب', 7), ('ك', 6), ('ن', 4), ('ر', 4), ('ت', 4), ('ش', 3)]`

## Worst examples

### retasy_train_005377_chunk_00
- gold: `منشر`
- pred: `والوساس`
- char_accuracy: 0.000
- edit_distance: 7
- lengths gold/pred: 4/7

### retasy_train_005444_chunk_00
- gold: `منشر`
- pred: `الرحم`
- char_accuracy: 0.000
- edit_distance: 5
- lengths gold/pred: 4/5

### retasy_train_005644_chunk_00
- gold: `منشر`
- pred: `منكالجنة`
- char_accuracy: 0.000
- edit_distance: 6
- lengths gold/pred: 4/8

### retasy_train_005677_chunk_00
- gold: `منشر`
- pred: `منالجناس`
- char_accuracy: 0.000
- edit_distance: 6
- lengths gold/pred: 4/8

### retasy_train_005699_chunk_00
- gold: `منشر`
- pred: `الرحمن`
- char_accuracy: 0.000
- edit_distance: 6
- lengths gold/pred: 4/6

### retasy_train_006092_chunk_00
- gold: `منشر`
- pred: `الرحن`
- char_accuracy: 0.000
- edit_distance: 5
- lengths gold/pred: 4/5

### retasy_train_006369_chunk_00
- gold: `منشر`
- pred: `انشاكند`
- char_accuracy: 0.000
- edit_distance: 5
- lengths gold/pred: 4/7

### retasy_train_005377_chunk_01
- gold: `الوسواس`
- pred: `واياكنوعبد`
- char_accuracy: 0.000
- edit_distance: 8
- lengths gold/pred: 7/10

### retasy_train_005610_chunk_01
- gold: `الوسواس`
- pred: `واياكنسعن`
- char_accuracy: 0.000
- edit_distance: 8
- lengths gold/pred: 7/9

### retasy_train_006314_chunk_01
- gold: `والناس`
- pred: `الرنبيمن`
- char_accuracy: 0.000
- edit_distance: 6
- lengths gold/pred: 6/8

### retasy_train_005361_chunk_00
- gold: `لااعبد`
- pred: `واياكنباس`
- char_accuracy: 0.000
- edit_distance: 6
- lengths gold/pred: 6/9

### retasy_train_006367_chunk_00
- gold: `لااعبد`
- pred: `واياكنان`
- char_accuracy: 0.000
- edit_distance: 6
- lengths gold/pred: 6/8

### retasy_train_000611_chunk_01
- gold: `ماتعبدون`
- pred: `والنياس`
- char_accuracy: 0.125
- edit_distance: 7
- lengths gold/pred: 8/7

### retasy_train_005361_chunk_01
- gold: `ماتعبدون`
- pred: `الكاناس`
- char_accuracy: 0.125
- edit_distance: 7
- lengths gold/pred: 8/7

### retasy_train_006006_chunk_01
- gold: `ماتعبدون`
- pred: `اياس`
- char_accuracy: 0.125
- edit_distance: 7
- lengths gold/pred: 8/4

### retasy_train_006367_chunk_01
- gold: `ماتعبدون`
- pred: `واكناس`
- char_accuracy: 0.125
- edit_distance: 7
- lengths gold/pred: 8/6

### retasy_train_006139_chunk_00
- gold: `لااعبد`
- pred: `لبدمن`
- char_accuracy: 0.167
- edit_distance: 5
- lengths gold/pred: 6/5

### retasy_train_005748_chunk_00
- gold: `ملكالناس`
- pred: `والنعبا`
- char_accuracy: 0.250
- edit_distance: 6
- lengths gold/pred: 8/7

### retasy_train_006466_chunk_00
- gold: `ملكالناس`
- pred: `الرحمن`
- char_accuracy: 0.250
- edit_distance: 6
- lengths gold/pred: 8/6

### retasy_train_002731_chunk_01
- gold: `ماتعبدون`
- pred: `التين`
- char_accuracy: 0.250
- edit_distance: 6
- lengths gold/pred: 8/5

### retasy_train_005677_chunk_01
- gold: `الوسواس`
- pred: `واكناس`
- char_accuracy: 0.286
- edit_distance: 5
- lengths gold/pred: 7/6

### retasy_train_006371_chunk_01
- gold: `والناس`
- pred: `عالرنعم`
- char_accuracy: 0.333
- edit_distance: 4
- lengths gold/pred: 6/7

### retasy_train_001310_chunk_00
- gold: `لااعبد`
- pred: `اليد`
- char_accuracy: 0.333
- edit_distance: 4
- lengths gold/pred: 6/4

### retasy_train_006006_chunk_00
- gold: `لااعبد`
- pred: `الاتعبدون`
- char_accuracy: 0.333
- edit_distance: 4
- lengths gold/pred: 6/9

### retasy_train_005364_chunk_00
- gold: `ملكالناس`
- pred: `النمن`
- char_accuracy: 0.375
- edit_distance: 5
- lengths gold/pred: 8/5

### retasy_train_005503_chunk_00
- gold: `ملكالناس`
- pred: `مالرحان`
- char_accuracy: 0.375
- edit_distance: 5
- lengths gold/pred: 8/7

### retasy_train_005563_chunk_00
- gold: `ملكالناس`
- pred: `الرحعباس`
- char_accuracy: 0.375
- edit_distance: 5
- lengths gold/pred: 8/8

### retasy_train_006177_chunk_00
- gold: `ملكالناس`
- pred: `اياكناة`
- char_accuracy: 0.375
- edit_distance: 5
- lengths gold/pred: 8/7

### retasy_train_000586_chunk_01
- gold: `الوسواس`
- pred: `والخناس`
- char_accuracy: 0.429
- edit_distance: 4
- lengths gold/pred: 7/7

### retasy_train_005444_chunk_01
- gold: `الوسواس`
- pred: `النا`
- char_accuracy: 0.429
- edit_distance: 4
- lengths gold/pred: 7/4


## Near misses

### retasy_train_001357_chunk_00
- gold: `ملكالناس`
- pred: `ملكاناس`
- char_accuracy: 0.875
- edit_distance: 1
- lengths gold/pred: 8/7

### retasy_train_005455_chunk_00
- gold: `ملكالناس`
- pred: `الكالناس`
- char_accuracy: 0.875
- edit_distance: 1
- lengths gold/pred: 8/8

### retasy_train_005724_chunk_00
- gold: `ملكالناس`
- pred: `الكالناس`
- char_accuracy: 0.875
- edit_distance: 1
- lengths gold/pred: 8/8

### retasy_train_000579_chunk_01
- gold: `ماتعبدون`
- pred: `ماعبدون`
- char_accuracy: 0.875
- edit_distance: 1
- lengths gold/pred: 8/7

### retasy_train_000662_chunk_01
- gold: `ماتعبدون`
- pred: `ماعبدون`
- char_accuracy: 0.875
- edit_distance: 1
- lengths gold/pred: 8/7

### retasy_train_000682_chunk_01
- gold: `ماتعبدون`
- pred: `ماتعبدتن`
- char_accuracy: 0.875
- edit_distance: 1
- lengths gold/pred: 8/8

### retasy_train_000888_chunk_01
- gold: `ماتعبدون`
- pred: `ماعبدون`
- char_accuracy: 0.875
- edit_distance: 1
- lengths gold/pred: 8/7

### retasy_train_001310_chunk_01
- gold: `ماتعبدون`
- pred: `ماعبدون`
- char_accuracy: 0.875
- edit_distance: 1
- lengths gold/pred: 8/7

### retasy_train_003573_chunk_01
- gold: `ماتعبدون`
- pred: `ماعبدون`
- char_accuracy: 0.875
- edit_distance: 1
- lengths gold/pred: 8/7

### retasy_train_004412_chunk_01
- gold: `ماتعبدون`
- pred: `ماتتعبدون`
- char_accuracy: 0.875
- edit_distance: 1
- lengths gold/pred: 8/9

### retasy_train_005147_chunk_01
- gold: `ماتعبدون`
- pred: `ماتعبدتن`
- char_accuracy: 0.875
- edit_distance: 1
- lengths gold/pred: 8/8

### retasy_train_006032_chunk_01
- gold: `ماتعبدون`
- pred: `ماعبدون`
- char_accuracy: 0.875
- edit_distance: 1
- lengths gold/pred: 8/7

### retasy_train_000373_chunk_01
- gold: `الوسواس`
- pred: `الوواس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_000494_chunk_01
- gold: `الوسواس`
- pred: `الوساس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_000501_chunk_01
- gold: `الوسواس`
- pred: `الوساس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_000724_chunk_01
- gold: `الوسواس`
- pred: `الوساس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_000843_chunk_01
- gold: `الوسواس`
- pred: `الوواس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_000918_chunk_01
- gold: `الوسواس`
- pred: `السواس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_001304_chunk_01
- gold: `الوسواس`
- pred: `الوساس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_001402_chunk_01
- gold: `الوسواس`
- pred: `الوساس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_001407_chunk_01
- gold: `الوسواس`
- pred: `الوساس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_001530_chunk_01
- gold: `الوسواس`
- pred: `الوساس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_001659_chunk_01
- gold: `الوسواس`
- pred: `الوواس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_001839_chunk_01
- gold: `الوسواس`
- pred: `الوساس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_001979_chunk_01
- gold: `الوسواس`
- pred: `الوساس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_002873_chunk_01
- gold: `الوسواس`
- pred: `الوساس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_003486_chunk_01
- gold: `الوسواس`
- pred: `الوواس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_004089_chunk_01
- gold: `الوسواس`
- pred: `الوساس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_004318_chunk_01
- gold: `الوسواس`
- pred: `الوساس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6

### retasy_train_004475_chunk_01
- gold: `الوسواس`
- pred: `الوساس`
- char_accuracy: 0.857
- edit_distance: 1
- lengths gold/pred: 7/6
