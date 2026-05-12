# Burst threshold ablation

- manifest: `data\manifests\retasy_burst_subset.jsonl`
- limit: 0

| threshold | rule | accuracy | precision | recall | f1 | FP | FN | none_acc | qalqalah_acc |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| argmax | argmax | 0.874 | 0.864 | 0.814 | 0.838 | 82 | 119 | 0.914 | 0.814 |
| 0.25 | qalqalah_probability_threshold | 0.820 | 0.704 | 0.947 | 0.808 | 254 | 34 | 0.735 | 0.947 |
| 0.30 | qalqalah_probability_threshold | 0.835 | 0.729 | 0.933 | 0.819 | 221 | 43 | 0.769 | 0.933 |
| 0.35 | qalqalah_probability_threshold | 0.846 | 0.752 | 0.917 | 0.827 | 193 | 53 | 0.799 | 0.917 |
| 0.40 | qalqalah_probability_threshold | 0.858 | 0.786 | 0.889 | 0.834 | 155 | 71 | 0.838 | 0.889 |
| 0.45 | qalqalah_probability_threshold | 0.868 | 0.820 | 0.858 | 0.839 | 120 | 91 | 0.875 | 0.858 |
| 0.50 | qalqalah_probability_threshold | 0.874 | 0.864 | 0.814 | 0.838 | 82 | 119 | 0.914 | 0.814 |
| 0.55 | qalqalah_probability_threshold | 0.868 | 0.875 | 0.781 | 0.825 | 71 | 140 | 0.926 | 0.781 |
| 0.60 | qalqalah_probability_threshold | 0.863 | 0.889 | 0.751 | 0.814 | 60 | 159 | 0.937 | 0.751 |
| 0.65 | qalqalah_probability_threshold | 0.855 | 0.906 | 0.710 | 0.796 | 47 | 185 | 0.951 | 0.710 |
| 0.70 | qalqalah_probability_threshold | 0.844 | 0.930 | 0.660 | 0.772 | 32 | 217 | 0.967 | 0.660 |

## Best

- best accuracy: `{'threshold': None, 'decision_rule': 'argmax', 'samples': 1597, 'accuracy': 0.8741390106449592, 'tn': 876, 'fp': 82, 'fn': 119, 'tp': 520, 'qalqalah_precision': 0.8637873754152824, 'qalqalah_recall': 0.8137715179968701, 'qalqalah_f1': 0.838033843674456, 'none_accuracy': 0.9144050104384134, 'qalqalah_accuracy': 0.8137715179968701}`
- best qalqalah F1: `{'threshold': 0.45, 'decision_rule': 'qalqalah_probability_threshold', 'samples': 1597, 'accuracy': 0.867877269881027, 'tn': 838, 'fp': 120, 'fn': 91, 'tp': 548, 'qalqalah_precision': 0.8203592814371258, 'qalqalah_recall': 0.8575899843505478, 'qalqalah_f1': 0.8385615914307575, 'none_accuracy': 0.8747390396659708, 'qalqalah_accuracy': 0.8575899843505478}`