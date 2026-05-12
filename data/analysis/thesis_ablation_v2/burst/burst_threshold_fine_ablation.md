# Burst threshold ablation

- manifest: `data\manifests\retasy_burst_subset.jsonl`
- limit: 0

| threshold | rule | accuracy | precision | recall | f1 | FP | FN | none_acc | qalqalah_acc |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| argmax | argmax | 0.874 | 0.864 | 0.814 | 0.838 | 82 | 119 | 0.914 | 0.814 |
| 0.42 | qalqalah_probability_threshold | 0.862 | 0.797 | 0.878 | 0.835 | 143 | 78 | 0.851 | 0.878 |
| 0.43 | qalqalah_probability_threshold | 0.864 | 0.804 | 0.873 | 0.837 | 136 | 81 | 0.858 | 0.873 |
| 0.44 | qalqalah_probability_threshold | 0.865 | 0.814 | 0.861 | 0.837 | 126 | 89 | 0.868 | 0.861 |
| 0.45 | qalqalah_probability_threshold | 0.868 | 0.820 | 0.858 | 0.839 | 120 | 91 | 0.875 | 0.858 |
| 0.46 | qalqalah_probability_threshold | 0.870 | 0.830 | 0.850 | 0.840 | 111 | 96 | 0.884 | 0.850 |
| 0.47 | qalqalah_probability_threshold | 0.875 | 0.843 | 0.847 | 0.845 | 101 | 98 | 0.895 | 0.847 |
| 0.48 | qalqalah_probability_threshold | 0.874 | 0.847 | 0.837 | 0.842 | 97 | 104 | 0.899 | 0.837 |
| 0.49 | qalqalah_probability_threshold | 0.874 | 0.854 | 0.826 | 0.840 | 90 | 111 | 0.906 | 0.826 |
| 0.50 | qalqalah_probability_threshold | 0.874 | 0.864 | 0.814 | 0.838 | 82 | 119 | 0.914 | 0.814 |

## Best

- best accuracy: `{'threshold': 0.47, 'decision_rule': 'qalqalah_probability_threshold', 'samples': 1597, 'accuracy': 0.8753913587977458, 'tn': 857, 'fp': 101, 'fn': 98, 'tp': 541, 'qalqalah_precision': 0.8426791277258567, 'qalqalah_recall': 0.8466353677621283, 'qalqalah_f1': 0.8446526151444185, 'none_accuracy': 0.894572025052192, 'qalqalah_accuracy': 0.8466353677621283}`
- best qalqalah F1: `{'threshold': 0.47, 'decision_rule': 'qalqalah_probability_threshold', 'samples': 1597, 'accuracy': 0.8753913587977458, 'tn': 857, 'fp': 101, 'fn': 98, 'tp': 541, 'qalqalah_precision': 0.8426791277258567, 'qalqalah_recall': 0.8466353677621283, 'qalqalah_f1': 0.8446526151444185, 'none_accuracy': 0.894572025052192, 'qalqalah_accuracy': 0.8466353677621283}`