# Content decoder tuning

- checkpoint: `checkpoints\content_v6c_old_expanded_full_vocab_hd96.pt`
- manifest: `data\manifests\retasy_content_chunks.jsonl`
- split: `val`
- split_mode: `text`
- samples: 417

## Results

| blank_penalty | exact | char_accuracy | edit_distance | avg_pred_len |
|---:|---:|---:|---:|---:|
| 0.00 | 0.700 | 0.888 | 0.772 | 6.24 |
| 0.40 | 0.705 | 0.890 | 0.748 | 6.29 |
| 0.80 | 0.688 | 0.887 | 0.765 | 6.31 |
| 1.00 | 0.688 | 0.888 | 0.760 | 6.32 |
| 1.20 | 0.688 | 0.888 | 0.763 | 6.32 |
| 1.40 | 0.691 | 0.889 | 0.755 | 6.35 |
| 1.60 | 0.693 | 0.889 | 0.755 | 6.36 |
| 1.80 | 0.688 | 0.887 | 0.760 | 6.38 |
| 2.00 | 0.688 | 0.887 | 0.763 | 6.39 |
| 2.40 | 0.662 | 0.882 | 0.796 | 6.44 |
| 2.80 | 0.638 | 0.877 | 0.830 | 6.49 |

## Best

- blank_penalty: 0.4
- exact_match: 0.705
- char_accuracy: 0.890
- edit_distance: 0.748

## Example errors

### retasy_train_004041_chunk_00
- gold: `منشر`
- pred: `منشرنس`
- char_accuracy: 0.500

### retasy_train_005377_chunk_00
- gold: `منشر`
- pred: `والوساس`
- char_accuracy: 0.000

### retasy_train_005444_chunk_00
- gold: `منشر`
- pred: `الرحم`
- char_accuracy: 0.000

### retasy_train_005644_chunk_00
- gold: `منشر`
- pred: `منكالجنة`
- char_accuracy: 0.000

### retasy_train_005677_chunk_00
- gold: `منشر`
- pred: `منالجناس`
- char_accuracy: 0.000

### retasy_train_005699_chunk_00
- gold: `منشر`
- pred: `الرحمن`
- char_accuracy: 0.000

### retasy_train_006092_chunk_00
- gold: `منشر`
- pred: `الرحن`
- char_accuracy: 0.000

### retasy_train_006369_chunk_00
- gold: `منشر`
- pred: `انشاكند`
- char_accuracy: 0.000

### retasy_train_000586_chunk_01
- gold: `الوسواس`
- pred: `والخناس`
- char_accuracy: 0.429

### retasy_train_005377_chunk_01
- gold: `الوسواس`
- pred: `واياكنوعبد`
- char_accuracy: 0.000

### retasy_train_005444_chunk_01
- gold: `الوسواس`
- pred: `النا`
- char_accuracy: 0.429

### retasy_train_005610_chunk_01
- gold: `الوسواس`
- pred: `واياكنسعن`
- char_accuracy: 0.000

### retasy_train_005677_chunk_01
- gold: `الوسواس`
- pred: `واكناس`
- char_accuracy: 0.286

### retasy_train_005699_chunk_01
- gold: `الوسواس`
- pred: `الحسعيان`
- char_accuracy: 0.429

### retasy_train_006092_chunk_01
- gold: `الوسواس`
- pred: `الرحاس`
- char_accuracy: 0.571

### retasy_train_006369_chunk_01
- gold: `الوسواس`
- pred: `والوسعيان`
- char_accuracy: 0.429

### retasy_train_000906_chunk_00
- gold: `ملكالناس`
- pred: `ملناس`
- char_accuracy: 0.625

### retasy_train_001920_chunk_00
- gold: `ملكالناس`
- pred: `منشدتر`
- char_accuracy: 0.125

### retasy_train_005364_chunk_00
- gold: `ملكالناس`
- pred: `النمن`
- char_accuracy: 0.375

### retasy_train_005416_chunk_00
- gold: `ملكالناس`
- pred: `والناس`
- char_accuracy: 0.625

### retasy_train_005503_chunk_00
- gold: `ملكالناس`
- pred: `مالرحان`
- char_accuracy: 0.375

### retasy_train_005508_chunk_00
- gold: `ملكالناس`
- pred: `والناس`
- char_accuracy: 0.625

### retasy_train_005525_chunk_00
- gold: `ملكالناس`
- pred: `وايالنعاس`
- char_accuracy: 0.500

### retasy_train_005563_chunk_00
- gold: `ملكالناس`
- pred: `الرحعباس`
- char_accuracy: 0.375

### retasy_train_005706_chunk_00
- gold: `ملكالناس`
- pred: `ملناس`
- char_accuracy: 0.625

### retasy_train_005740_chunk_00
- gold: `ملكالناس`
- pred: `والناس`
- char_accuracy: 0.625

### retasy_train_005748_chunk_00
- gold: `ملكالناس`
- pred: `والنعبا`
- char_accuracy: 0.250

### retasy_train_005998_chunk_00
- gold: `ملكالناس`
- pred: `مناناس`
- char_accuracy: 0.625

### retasy_train_006177_chunk_00
- gold: `ملكالناس`
- pred: `اياكناة`
- char_accuracy: 0.375

### retasy_train_006466_chunk_00
- gold: `ملكالناس`
- pred: `الرحمن`
- char_accuracy: 0.250
