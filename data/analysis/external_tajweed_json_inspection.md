# External Tajweed JSON inspection

Found JSON files: 1201

## Likely Quran/Tajweed JSON files

- `data\analysis\ablations\ayah_expected_ctc_v2_bp12_val_summary.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\ablations\ayah_strict_free_decode_v2_bp12_val_summary.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\ablations\content_v6c_self_diagnostics.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\ablations\modular_content_beam_bp04.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\ablations\modular_content_eval_lexicon_bp04.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\ablations\modular_content_reciter_split_burst047.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\ablations\modular_default_transition_argmax_burst047_with_examples.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\ablations\routing_profiles_retasy_all.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\ablations\routing_profiles_retasy_calibration.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\ablations\routing_profiles_v5_mixed_val.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\ayah_content_inference_sample0.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\batch_ayah_content_v2_bp12_val_smoke50_summary.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\batch_ayah_content_v2_bp12_val_summary.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\chunked_content_failures.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\chunked_content_failures_improved.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\chunked_content_hardcase_failures.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\chunked_content_hardcases.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\chunked_content_open_hd96_failures_textsplit.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\chunked_content_textsplit_hd96_failures.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\content_ayah_hf_v1_blank_penalty_tune.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\content_ayah_hf_v1_fullverse_buckets_bp12_limit500.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\content_ayah_hf_v1_fullverse_buckets_limit500.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\content_ayah_hf_v2_balanced_fullverse_buckets_bp12_limit500.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\content_ayah_v1_vs_v2_reciter_breakdown.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\content_failures.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\content_prediction_sample_0.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\content_regression_balanced_only_limit100.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\content_regression_safe_v1_limit100.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\content_regression_tiny_aux_v2_limit100.json` score=2 arabic=True html=False tajweed_terms=True
- `data\analysis\content_v6c_decoder_blank_penalty_tune.json` score=2 arabic=True html=False tajweed_terms=True

## Top HTML classes


## Top style colors


## Structure previews

### `data\analysis\ablations\ayah_expected_ctc_v2_bp12_val_summary.json`

```text
$: dict keys=['manifest', 'split', 'limit', 'checkpoint', 'decoder_config', 'blank_penalty', 'overall', 'by_reciter', 'best_expected_text', 'worst_expected_text']
$.manifest: str 'C:\\Users\\anis\\Desktop\\tajweed-modular-assessment\\data\\manifests\\content_v6a_short_hf_ayah_r1_hf_ayah_clean_all.jsonl'
$.split: str 'val'
$.limit: int 0
$.checkpoint: str 'C:\\Users\\anis\\Desktop\\tajweed-modular-assessment\\checkpoints\\content_ayah_hf_v2_balanced_hd96.pt'
$.decoder_config: str 'C:\\Users\\anis\\Desktop\\tajweed-modular-assessment\\configs\\content_ayah_decoder_bp12.json'
$.blank_penalty: float 1.2
$.overall: dict keys=['samples', 'free_exact_rate', 'expected_text_accepted_rate', 'expected_text_strong_review_rate', 'expected_text_plausible_review_rate', 'avg_free_char_accuracy', 'avg_expected_ctc_loss_per_char', 'avg_expected_ctc_confidence', 'verdict_counts']
$.overall.samples: int 448
$.overall.free_exact_rate: float 0.0625
$.overall.expected_text_accepted_rate: float 0.11383928571428571
$.overall.expected_text_strong_review_rate: float 0.40401785714285715
$.overall.expected_text_plausible_review_rate: float 0.24330357142857142
$.overall.avg_free_char_accuracy: float 0.7621132140627868
$.overall.avg_expected_ctc_loss_per_char: float 0.8407779494098053
$.overall.avg_expected_ctc_confidence: float 0.5052350407191526
$.by_reciter: dict keys=['abdul_basit_murattal', 'abdullaah_3awwaad_al_juhaynee', 'abdullah_basfar', 'abdurrahmaan_as_sudais', 'abu_bakr_ash_shaatree', 'alafasy', 'ali_jaber', 'banna', 'ghamadi', 'husary_mujawwad', 'hussary.teacher', 'ibrahim_akhdar', 'minshawy_mujawwad', 'muhsin_al_qasim', 'saood_ash_shuraym', 'warsh_husary', 'warsh_yassin']
$.by_reciter.abdul_basit_murattal: dict keys=['samples', 'free_exact_rate', 'expected_text_accepted_rate', 'expected_text_strong_review_rate', 'expected_text_plausible_review_rate', 'avg_free_char_accuracy', 'avg_expected_ctc_loss_per_char', 'avg_expected_ctc_confidence', 'verdict_counts']
$.by_reciter.abdul_basit_murattal.samples: int 46
$.by_reciter.abdul_basit_murattal.free_exact_rate: float 0.08695652173913043
$.by_reciter.abdul_basit_murattal.expected_text_accepted_rate: float 0.21739130434782608
$.by_reciter.abdul_basit_murattal.expected_text_strong_review_rate: float 0.5652173913043478
$.by_reciter.abdul_basit_murattal.expected_text_plausible_review_rate: float 0.1956521739130435
$.by_reciter.abdul_basit_murattal.avg_free_char_accuracy: float 0.8710239678677788
$.by_reciter.abdul_basit_murattal.avg_expected_ctc_loss_per_char: float 0.47669609152001674
$.by_reciter.abdul_basit_murattal.avg_expected_ctc_confidence: float 0.6332985018108654
$.by_reciter.abdullaah_3awwaad_al_juhaynee: dict keys=['samples', 'free_exact_rate', 'expected_text_accepted_rate', 'expected_text_strong_review_rate', 'expected_text_plausible_review_rate', 'avg_free_char_accuracy', 'avg_expected_ctc_loss_per_char', 'avg_expected_ctc_confidence', 'verdict_counts']
$.by_reciter.abdullaah_3awwaad_al_juhaynee.samples: int 41
$.by_reciter.abdullaah_3awwaad_al_juhaynee.free_exact_rate: float 0.0
$.by_reciter.abdullaah_3awwaad_al_juhaynee.expected_text_accepted_rate: float 0.0
$.by_reciter.abdullaah_3awwaad_al_juhaynee.expected_text_strong_review_rate: float 0.0
$.by_reciter.abdullaah_3awwaad_al_juhaynee.expected_text_plausible_review_rate: float 0.0
$.by_reciter.abdullaah_3awwaad_al_juhaynee.avg_free_char_accuracy: float 0.31933262176264293
$.by_reciter.abdullaah_3awwaad_al_juhaynee.avg_expected_ctc_loss_per_char: float 2.467960058186919
$.by_reciter.abdullaah_3awwaad_al_juhaynee.avg_expected_ctc_confidence: float 0.08986160555314787
$.by_reciter.abdullah_basfar: dict keys=['samples', 'free_exact_rate', 'expected_text_accepted_rate', 'expected_text_strong_review_rate', 'expected_text_plausible_review_rate', 'avg_free_char_accuracy', 'avg_expected_ctc_loss_per_char', 'avg_expected_ctc_confidence', 'verdict_counts']
$.by_reciter.abdullah_basfar.samples: int 11
$.by_reciter.abdullah_basfar.free_exact_rate: float 0.0
$.by_reciter.abdullah_basfar.expected_text_accepted_rate: float 0.09090909090909091
$.by_reciter.abdullah_basfar.expected_text_strong_review_rate: float 0.6363636363636364
$.by_reciter.abdullah_basfar.expected_text_plausible_review_rate: float 0.2727272727272727
$.by_reciter.abdullah_basfar.avg_free_char_accuracy: float 0.8482410737453961
$.by_reciter.abdullah_basfar.avg_expected_ctc_loss_per_char: float 0.5295646562629733
$.by_reciter.abdullah_basfar.avg_expected_ctc_confidence: float 0.5976074935157956
$.by_reciter.abdurrahmaan_as_sudais: dict keys=['samples', 'free_exact_rate', 'expected_text_accepted_rate', 'expected_text_strong_review_rate', 'expected_text_plausible_review_rate', 'avg_free_char_accuracy', 'avg_expected_ctc_loss_per_char', 'avg_expected_ctc_confidence', 'verdict_counts']
$.by_reciter.abdurrahmaan_as_sudais.samples: int 11
$.by_reciter.abdurrahmaan_as_sudais.free_exact_rate: float 0.0
$.by_reciter.abdurrahmaan_as_sudais.expected_text_accepted_rate: float 0.0
$.by_reciter.abdurrahmaan_as_sudais.expected_text_strong_review_rate: float 0.09090909090909091
$.by_reciter.abdurrahmaan_as_sudais.expected_text_plausible_review_rate: float 0.5454545454545454
$.by_reciter.abdurrahmaan_as_sudais.avg_free_char_accuracy: float 0.723851538733988
$.by_reciter.abdurrahmaan_as_sudais.avg_expected_ctc_loss_per_char: float 0.9828123832651702
$.by_reciter.abdurrahmaan_as_sudais.avg_expected_ctc_confidence: float 0.3983741455575462
$.by_reciter.abu_bakr_ash_shaatree: dict keys=['samples', 'free_exact_rate', 'expected_text_accepted_rate', 'expected_text_strong_review_rate', 'expected_text_plausible_review_rate', 'avg_free_char_accuracy', 'avg_expected_ctc_loss_per_char', 'avg_expected_ctc_confidence', 'verdict_counts']
$.by_reciter.abu_bakr_ash_shaatree.samples: int 56
$.by_reciter.abu_bakr_ash_shaatree.free_exact_rate: float 0.0
$.by_reciter.abu_bakr_ash_shaatree.expected_text_accepted_rate: float 0.017857142857142856
$.by_reciter.abu_bakr_ash_shaatree.expected_text_strong_review_rate: float 0.17857142857142858
$.by_reciter.abu_bakr_ash_shaatree.expected_text_plausible_review_rate: float 0.4107142857142857
$.by_reciter.abu_bakr_ash_shaatree.avg_free_char_accuracy: float 0.7072291928304015
$.by_reciter.abu_bakr_ash_shaatree.avg_expected_ctc_loss_per_char: float 0.9779710502991655
$.by_reciter.abu_bakr_ash_shaatree.avg_expected_ctc_confidence: float 0.4004150322940224
$.by_reciter.alafasy: dict keys=['samples', 'free_exact_rate', 'expected_text_accepted_rate', 'expected_text_strong_review_rate', 'expected_text_plausible_review_rate', 'avg_free_char_accuracy', 'avg_expected_ctc_loss_per_char', 'avg_expected_ctc_confidence', 'verdict_counts']
$.by_reciter.alafasy.samples: int 23
$.by_reciter.alafasy.free_exact_rate: float 0.13043478260869565
$.by_reciter.alafasy.expected_text_accepted_rate: float 0.21739130434782608
$.by_reciter.alafasy.expected_text_strong_review_rate: float 0.6521739130434783
$.by_reciter.alafasy.expected_text_plausible_review_rate: float 0.13043478260869565
$.by_reciter.alafasy.avg_free_char_accuracy: float 0.8833791517982318
$.by_reciter.alafasy.avg_expected_ctc_loss_per_char: float 0.45649296168586717
$.by_reciter.alafasy.avg_expected_ctc_confidence: float 0.6435852625581857
$.by_reciter.ali_jaber: dict keys=['samples', 'free_exact_rate', 'expected_text_accepted_rate', 'expected_text_strong_review_rate', 'expected_text_plausible_review_rate', 'avg_free_char_accuracy', 'avg_expected_ctc_loss_per_char', 'avg_expected_ctc_confidence', 'verdict_counts']
$.by_reciter.ali_jaber.samples: int 20
$.by_reciter.ali_jaber.free_exact_rate: float 0.25
$.by_reciter.ali_jaber.expected_text_accepted_rate: float 0.25
$.by_reciter.ali_jaber.expected_text_strong_review_rate: float 0.55
$.by_reciter.ali_jaber.expected_text_plausible_review_rate: float 0.15
$.by_reciter.ali_jaber.avg_free_char_accuracy: float 0.8917378734659511
$.by_reciter.ali_jaber.avg_expected_ctc_loss_per_char: float 0.439582729708455
$.by_reciter.ali_jaber.avg_expected_ctc_confidence: float 0.670118763286619
$.by_reciter.banna: dict keys=['samples', 'free_exact_rate', 'expected_text_accepted_rate', 'expected_text_strong_review_rate', 'expected_text_plausible_review_rate', 'avg_free_char_accuracy', 'avg_expected_ctc_loss_per_char', 'avg_expected_ctc_confidence', 'verdict_counts']
$.by_reciter.banna.samples: int 31
$.by_reciter.banna.free_exact_rate: float 0.06451612903225806
$.by_reciter.banna.expected_text_accepted_rate: float 0.0967741935483871
$.by_reciter.banna.expected_text_strong_review_rate: float 0.5806451612903226
$.by_reciter.banna.expected_text_plausible_review_rate: float 0.3225806451612903
$.by_reciter.banna.avg_free_char_accuracy: float 0.8537753429124239
$.by_reciter.banna.avg_expected_ctc_loss_per_char: float 0.529916661981202
$.by_reciter.banna.avg_expected_ctc_confidence: float 0.6009329332883159
```

### `data\analysis\ablations\ayah_strict_free_decode_v2_bp12_val_summary.json`

```text
$: dict keys=['manifest', 'split', 'limit', 'checkpoint', 'decoder_config', 'blank_penalty', 'overall', 'by_reciter', 'worst_examples', 'best_examples']
$.manifest: str 'C:\\Users\\anis\\Desktop\\tajweed-modular-assessment\\data\\manifests\\content_v6a_short_hf_ayah_r1_hf_ayah_clean_all.jsonl'
$.split: str 'val'
$.limit: int 0
$.checkpoint: str 'C:\\Users\\anis\\Desktop\\tajweed-modular-assessment\\checkpoints\\content_ayah_hf_v2_balanced_hd96.pt'
$.decoder_config: str 'C:\\Users\\anis\\Desktop\\tajweed-modular-assessment\\configs\\content_ayah_decoder_bp12.json'
$.blank_penalty: float 1.2
$.overall: dict keys=['samples', 'avg_score', 'avg_char_accuracy', 'avg_edit_distance', 'avg_edit_rate', 'exact_rate', 'accepted_rate', 'acceptance_counts', 'quality_counts']
$.overall.samples: int 448
$.overall.avg_score: float 76.21142857142857
$.overall.avg_char_accuracy: float 0.7621132140627868
$.overall.avg_edit_distance: float 6.446428571428571
$.overall.avg_edit_rate: float 0.23788678593721327
$.overall.exact_rate: float 0.0625
$.overall.accepted_rate: float 0.0625
$.overall.acceptance_counts: dict keys=['accepted_exact', 'not_accepted']
$.overall.acceptance_counts.accepted_exact: int 28
$.overall.acceptance_counts.not_accepted: int 420
$.by_reciter: dict keys=['abdul_basit_murattal', 'abdullaah_3awwaad_al_juhaynee', 'abdullah_basfar', 'abdurrahmaan_as_sudais', 'abu_bakr_ash_shaatree', 'alafasy', 'ali_jaber', 'banna', 'ghamadi', 'husary_mujawwad', 'hussary.teacher', 'ibrahim_akhdar', 'minshawy_mujawwad', 'muhsin_al_qasim', 'saood_ash_shuraym', 'warsh_husary', 'warsh_yassin']
$.by_reciter.abdul_basit_murattal: dict keys=['samples', 'avg_score', 'avg_char_accuracy', 'avg_edit_distance', 'avg_edit_rate', 'exact_rate', 'accepted_rate', 'acceptance_counts', 'quality_counts']
$.by_reciter.abdul_basit_murattal.samples: int 46
$.by_reciter.abdul_basit_murattal.avg_score: float 87.10217391304347
$.by_reciter.abdul_basit_murattal.avg_char_accuracy: float 0.8710239678677788
$.by_reciter.abdul_basit_murattal.avg_edit_distance: float 4.304347826086956
$.by_reciter.abdul_basit_murattal.avg_edit_rate: float 0.12897603213222114
$.by_reciter.abdul_basit_murattal.exact_rate: float 0.08695652173913043
$.by_reciter.abdul_basit_murattal.accepted_rate: float 0.08695652173913043
$.by_reciter.abdul_basit_murattal.acceptance_counts: dict keys=['not_accepted', 'accepted_exact']
$.by_reciter.abdul_basit_murattal.acceptance_counts.not_accepted: int 42
$.by_reciter.abdul_basit_murattal.acceptance_counts.accepted_exact: int 4
$.by_reciter.abdullaah_3awwaad_al_juhaynee: dict keys=['samples', 'avg_score', 'avg_char_accuracy', 'avg_edit_distance', 'avg_edit_rate', 'exact_rate', 'accepted_rate', 'acceptance_counts', 'quality_counts']
$.by_reciter.abdullaah_3awwaad_al_juhaynee.samples: int 41
$.by_reciter.abdullaah_3awwaad_al_juhaynee.avg_score: float 31.933170731707317
$.by_reciter.abdullaah_3awwaad_al_juhaynee.avg_char_accuracy: float 0.31933262176264293
$.by_reciter.abdullaah_3awwaad_al_juhaynee.avg_edit_distance: float 17.5609756097561
$.by_reciter.abdullaah_3awwaad_al_juhaynee.avg_edit_rate: float 0.6806673782373571
$.by_reciter.abdullaah_3awwaad_al_juhaynee.exact_rate: float 0.0
$.by_reciter.abdullaah_3awwaad_al_juhaynee.accepted_rate: float 0.0
$.by_reciter.abdullaah_3awwaad_al_juhaynee.acceptance_counts: dict keys=['not_accepted']
$.by_reciter.abdullaah_3awwaad_al_juhaynee.acceptance_counts.not_accepted: int 41
$.by_reciter.abdullah_basfar: dict keys=['samples', 'avg_score', 'avg_char_accuracy', 'avg_edit_distance', 'avg_edit_rate', 'exact_rate', 'accepted_rate', 'acceptance_counts', 'quality_counts']
$.by_reciter.abdullah_basfar.samples: int 11
$.by_reciter.abdullah_basfar.avg_score: float 84.82454545454544
$.by_reciter.abdullah_basfar.avg_char_accuracy: float 0.8482410737453961
$.by_reciter.abdullah_basfar.avg_edit_distance: float 5.909090909090909
$.by_reciter.abdullah_basfar.avg_edit_rate: float 0.15175892625460377
$.by_reciter.abdullah_basfar.exact_rate: float 0.0
$.by_reciter.abdullah_basfar.accepted_rate: float 0.0
$.by_reciter.abdullah_basfar.acceptance_counts: dict keys=['not_accepted']
$.by_reciter.abdullah_basfar.acceptance_counts.not_accepted: int 11
$.by_reciter.abdurrahmaan_as_sudais: dict keys=['samples', 'avg_score', 'avg_char_accuracy', 'avg_edit_distance', 'avg_edit_rate', 'exact_rate', 'accepted_rate', 'acceptance_counts', 'quality_counts']
$.by_reciter.abdurrahmaan_as_sudais.samples: int 11
$.by_reciter.abdurrahmaan_as_sudais.avg_score: float 72.38636363636364
$.by_reciter.abdurrahmaan_as_sudais.avg_char_accuracy: float 0.723851538733988
$.by_reciter.abdurrahmaan_as_sudais.avg_edit_distance: float 11.0
$.by_reciter.abdurrahmaan_as_sudais.avg_edit_rate: float 0.27614846126601206
$.by_reciter.abdurrahmaan_as_sudais.exact_rate: float 0.0
$.by_reciter.abdurrahmaan_as_sudais.accepted_rate: float 0.0
$.by_reciter.abdurrahmaan_as_sudais.acceptance_counts: dict keys=['not_accepted']
$.by_reciter.abdurrahmaan_as_sudais.acceptance_counts.not_accepted: int 11
$.by_reciter.abu_bakr_ash_shaatree: dict keys=['samples', 'avg_score', 'avg_char_accuracy', 'avg_edit_distance', 'avg_edit_rate', 'exact_rate', 'accepted_rate', 'acceptance_counts', 'quality_counts']
$.by_reciter.abu_bakr_ash_shaatree.samples: int 56
$.by_reciter.abu_bakr_ash_shaatree.avg_score: float 70.72232142857142
$.by_reciter.abu_bakr_ash_shaatree.avg_char_accuracy: float 0.7072291928304015
$.by_reciter.abu_bakr_ash_shaatree.avg_edit_distance: float 4.839285714285714
$.by_reciter.abu_bakr_ash_shaatree.avg_edit_rate: float 0.2927708071695984
$.by_reciter.abu_bakr_ash_shaatree.exact_rate: float 0.0
$.by_reciter.abu_bakr_ash_shaatree.accepted_rate: float 0.0
$.by_reciter.abu_bakr_ash_shaatree.acceptance_counts: dict keys=['not_accepted']
$.by_reciter.abu_bakr_ash_shaatree.acceptance_counts.not_accepted: int 56
$.by_reciter.alafasy: dict keys=['samples', 'avg_score', 'avg_char_accuracy', 'avg_edit_distance', 'avg_edit_rate', 'exact_rate', 'accepted_rate', 'acceptance_counts', 'quality_counts']
$.by_reciter.alafasy.samples: int 23
$.by_reciter.alafasy.avg_score: float 88.3391304347826
$.by_reciter.alafasy.avg_char_accuracy: float 0.8833791517982318
$.by_reciter.alafasy.avg_edit_distance: float 4.043478260869565
$.by_reciter.alafasy.avg_edit_rate: float 0.11662084820176828
$.by_reciter.alafasy.exact_rate: float 0.13043478260869565
$.by_reciter.alafasy.accepted_rate: float 0.13043478260869565
$.by_reciter.alafasy.acceptance_counts: dict keys=['not_accepted', 'accepted_exact']
$.by_reciter.alafasy.acceptance_counts.not_accepted: int 20
$.by_reciter.alafasy.acceptance_counts.accepted_exact: int 3
$.by_reciter.ali_jaber: dict keys=['samples', 'avg_score', 'avg_char_accuracy', 'avg_edit_distance', 'avg_edit_rate', 'exact_rate', 'accepted_rate', 'acceptance_counts', 'quality_counts']
$.by_reciter.ali_jaber.samples: int 20
$.by_reciter.ali_jaber.avg_score: float 89.174
$.by_reciter.ali_jaber.avg_char_accuracy: float 0.8917378734659511
$.by_reciter.ali_jaber.avg_edit_distance: float 3.95
$.by_reciter.ali_jaber.avg_edit_rate: float 0.1082621265340488
$.by_reciter.ali_jaber.exact_rate: float 0.25
$.by_reciter.ali_jaber.accepted_rate: float 0.25
$.by_reciter.ali_jaber.acceptance_counts: dict keys=['not_accepted', 'accepted_exact']
$.by_reciter.ali_jaber.acceptance_counts.not_accepted: int 15
$.by_reciter.ali_jaber.acceptance_counts.accepted_exact: int 5
$.by_reciter.banna: dict keys=['samples', 'avg_score', 'avg_char_accuracy', 'avg_edit_distance', 'avg_edit_rate', 'exact_rate', 'accepted_rate', 'acceptance_counts', 'quality_counts']
$.by_reciter.banna.samples: int 31
$.by_reciter.banna.avg_score: float 85.37806451612903
$.by_reciter.banna.avg_char_accuracy: float 0.8537753429124239
$.by_reciter.banna.avg_edit_distance: float 5.645161290322581
$.by_reciter.banna.avg_edit_rate: float 0.1462246570875762
$.by_reciter.banna.exact_rate: float 0.06451612903225806
$.by_reciter.banna.accepted_rate: float 0.06451612903225806
$.by_reciter.banna.acceptance_counts: dict keys=['not_accepted', 'accepted_exact']
$.by_reciter.banna.acceptance_counts.not_accepted: int 29
$.by_reciter.banna.acceptance_counts.accepted_exact: int 2
```

### `data\analysis\ablations\content_v6c_self_diagnostics.json`

```text
$: dict keys=['checkpoint', 'manifest', 'split', 'decoder_config', 'blank_penalty', 'overall', 'examples', 'worst_examples']
$.checkpoint: str 'C:\\Users\\anis\\Desktop\\tajweed-modular-assessment\\checkpoints\\content_v6c_old_expanded_full_vocab_hd96.pt'
$.manifest: str 'C:\\Users\\anis\\Desktop\\tajweed-modular-assessment\\data\\manifests\\retasy_content_chunks.jsonl'
$.split: str 'val'
$.decoder_config: str 'C:\\Users\\anis\\Desktop\\tajweed-modular-assessment\\checkpoints\\content_chunked_decoder_open_hd96_v6c_tuned.json'
$.blank_penalty: float 0.4
$.overall: dict keys=['samples', 'exact_match', 'char_accuracy', 'edit_distance', 'avg_gold_len', 'avg_pred_len']
$.overall.samples: int 417
$.overall.exact_match: float 0.8968824940047961
$.overall.char_accuracy: float 0.9776931701392133
$.overall.edit_distance: float 0.15827338129496402
$.overall.avg_gold_len: float 7.2398081534772185
$.overall.avg_pred_len: float 7.143884892086331
$.examples: list len=417
$.examples[0]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.examples[0].id: str 'retasy_train_000020_chunk_00'
$.examples[0].gold: str 'الرحمن'
$.examples[0].pred: str 'الرحمن'
$.examples[0].exact: bool True
$.examples[0].char_accuracy: float 1.0
$.examples[0].edit_distance: int 0
$.examples[0].gold_len: int 6
$.examples[0].pred_len: int 6
$.examples[1]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.examples[1].id: str 'retasy_train_000020_chunk_01'
$.examples[1].gold: str 'الرحيم'
$.examples[1].pred: str 'الرحيم'
$.examples[1].exact: bool True
$.examples[1].char_accuracy: float 1.0
$.examples[1].edit_distance: int 0
$.examples[1].gold_len: int 6
$.examples[1].pred_len: int 6
$.examples[2]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.examples[2].id: str 'retasy_train_000029_chunk_00'
$.examples[2].gold: str 'منشر'
$.examples[2].pred: str 'منشر'
$.examples[2].exact: bool True
$.examples[2].char_accuracy: float 1.0
$.examples[2].edit_distance: int 0
$.examples[2].gold_len: int 4
$.examples[2].pred_len: int 4
$.worst_examples: list len=50
$.worst_examples[0]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.worst_examples[0].id: str 'retasy_train_000611_chunk_01'
$.worst_examples[0].gold: str 'ماتعبدون'
$.worst_examples[0].pred: str 'والنياس'
$.worst_examples[0].exact: bool False
$.worst_examples[0].char_accuracy: float 0.125
$.worst_examples[0].edit_distance: int 7
$.worst_examples[0].gold_len: int 8
$.worst_examples[0].pred_len: int 7
$.worst_examples[1]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.worst_examples[1].id: str 'retasy_train_001129_chunk_00'
$.worst_examples[1].gold: str 'الرحمن'
$.worst_examples[1].pred: str 'الخناس'
$.worst_examples[1].exact: bool False
$.worst_examples[1].char_accuracy: float 0.33333333333333337
$.worst_examples[1].edit_distance: int 4
$.worst_examples[1].gold_len: int 6
$.worst_examples[1].pred_len: int 6
$.worst_examples[2]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.worst_examples[2].id: str 'retasy_train_001310_chunk_00'
$.worst_examples[2].gold: str 'لااعبد'
$.worst_examples[2].pred: str 'اليد'
$.worst_examples[2].exact: bool False
$.worst_examples[2].char_accuracy: float 0.33333333333333337
$.worst_examples[2].edit_distance: int 4
$.worst_examples[2].gold_len: int 6
$.worst_examples[2].pred_len: int 4
```

### `data\analysis\ablations\modular_content_beam_bp04.json`

```text
$: dict keys=['duration', 'transition', 'burst', 'content', 'content_reference_full_verse', 'weighted_scoring', 'duration_checkpoint', 'transition_checkpoint', 'ablation_flags']
$.duration: dict keys=['samples', 'route_counts', 'total_positions', 'correct_positions', 'accuracy', 'rule_summary', 'hybrid_support']
$.duration.samples: int 973
$.duration.route_counts: dict keys=['duration']
$.duration.route_counts.duration: int 973
$.duration.total_positions: int 1646
$.duration.correct_positions: int 1634
$.duration.accuracy: float 0.9927095990279465
$.duration.rule_summary: dict keys=['ghunnah', 'madd']
$.duration.rule_summary.ghunnah: dict keys=['total', 'correct', 'accuracy']
$.duration.rule_summary.ghunnah.total: int 364
$.duration.rule_summary.ghunnah.correct: int 358
$.duration.rule_summary.ghunnah.accuracy: float 0.9835164835164835
$.duration.rule_summary.madd: dict keys=['total', 'correct', 'accuracy']
$.duration.rule_summary.madd.total: int 1282
$.duration.rule_summary.madd.correct: int 1276
$.duration.rule_summary.madd.accuracy: float 0.9953198127925117
$.duration.hybrid_support: dict keys=['localized_available', 'localized_same_as_sequence', 'localized_same_rate', 'localized_supports_gold', 'localized_supports_gold_rate', 'localized_supports_sequence', 'localized_supports_sequence_rate', 'localized_disagrees_with_sequence', 'gold_supported_by_class', 'sequence_supported_by_class']
$.duration.hybrid_support.localized_available: int 1646
$.duration.hybrid_support.localized_same_as_sequence: int 1607
$.duration.hybrid_support.localized_same_rate: float 0.9763061968408262
$.duration.hybrid_support.localized_supports_gold: int 1600
$.duration.hybrid_support.localized_supports_gold_rate: float 0.9720534629404617
$.duration.hybrid_support.localized_supports_sequence: int 1607
$.duration.hybrid_support.localized_supports_sequence_rate: float 0.9763061968408262
$.duration.hybrid_support.localized_disagrees_with_sequence: int 39
$.transition: dict keys=['available', 'samples', 'accuracy', 'confusion_matrix', 'class_summary', 'hybrid_support']
$.transition.available: bool True
$.transition.samples: int 690
$.transition.accuracy: float 0.9101449275362319
$.transition.confusion_matrix: list len=3
$.transition.confusion_matrix[0]: list len=3
$.transition.confusion_matrix[0][0]: int 381
$.transition.confusion_matrix[0][1]: int 22
$.transition.confusion_matrix[0][2]: int 11
$.transition.confusion_matrix[1]: list len=3
$.transition.confusion_matrix[1][0]: int 21
$.transition.confusion_matrix[1][1]: int 205
$.transition.confusion_matrix[1][2]: int 1
$.transition.confusion_matrix[2]: list len=3
$.transition.confusion_matrix[2][0]: int 7
$.transition.confusion_matrix[2][1]: int 0
$.transition.confusion_matrix[2][2]: int 42
$.transition.class_summary: dict keys=['none', 'ikhfa', 'idgham']
$.transition.class_summary.none: dict keys=['total', 'correct', 'accuracy']
$.transition.class_summary.none.total: int 414
$.transition.class_summary.none.correct: int 381
$.transition.class_summary.none.accuracy: float 0.9202898550724637
$.transition.class_summary.ikhfa: dict keys=['total', 'correct', 'accuracy']
$.transition.class_summary.ikhfa.total: int 227
$.transition.class_summary.ikhfa.correct: int 205
$.transition.class_summary.ikhfa.accuracy: float 0.9030837004405287
$.transition.class_summary.idgham: dict keys=['total', 'correct', 'accuracy']
$.transition.class_summary.idgham.total: int 49
$.transition.class_summary.idgham.correct: int 42
$.transition.class_summary.idgham.accuracy: float 0.8571428571428571
$.transition.hybrid_support: dict keys=['localized_available', 'localized_same_as_whole_verse', 'localized_same_rate', 'localized_supports_gold', 'localized_supports_gold_rate', 'localized_supports_whole_verse', 'localized_supports_whole_verse_rate', 'localized_disagrees_with_whole_verse', 'gold_supported_by_class', 'whole_verse_supported_by_class']
$.transition.hybrid_support.localized_available: int 308
$.transition.hybrid_support.localized_same_as_whole_verse: int 268
$.transition.hybrid_support.localized_same_rate: float 0.8701298701298701
$.transition.hybrid_support.localized_supports_gold: int 112
$.transition.hybrid_support.localized_supports_gold_rate: float 0.36363636363636365
$.transition.hybrid_support.localized_supports_whole_verse: int 104
$.transition.hybrid_support.localized_supports_whole_verse_rate: float 0.33766233766233766
$.transition.hybrid_support.localized_disagrees_with_whole_verse: int 40
$.burst: dict keys=['available', 'samples', 'accuracy', 'confusion_matrix', 'class_summary', 'burst_threshold', 'decision_rule']
$.burst.available: bool True
$.burst.samples: int 1597
$.burst.accuracy: float 0.8753913587977458
$.burst.confusion_matrix: list len=2
$.burst.confusion_matrix[0]: list len=2
$.burst.confusion_matrix[0][0]: int 857
$.burst.confusion_matrix[0][1]: int 101
$.burst.confusion_matrix[1]: list len=2
$.burst.confusion_matrix[1][0]: int 98
$.burst.confusion_matrix[1][1]: int 541
$.burst.class_summary: dict keys=['none', 'qalqalah']
$.burst.class_summary.none: dict keys=['total', 'correct', 'accuracy']
$.burst.class_summary.none.total: int 958
$.burst.class_summary.none.correct: int 857
$.burst.class_summary.none.accuracy: float 0.894572025052192
$.burst.class_summary.qalqalah: dict keys=['total', 'correct', 'accuracy']
$.burst.class_summary.qalqalah.total: int 639
$.burst.class_summary.qalqalah.correct: int 541
$.burst.class_summary.qalqalah.accuracy: float 0.8466353677621283
$.burst.burst_threshold: float 0.47
$.burst.decision_rule: str 'qalqalah_probability_threshold'
$.content: dict keys=['available', 'mode', 'samples', 'split', 'split_mode', 'examples', 'worst_examples', 'decoder', 'exact_match', 'char_accuracy', 'edit_distance']
$.content.available: bool True
$.content.mode: str 'chunked'
$.content.samples: int 417
$.content.split: str 'val'
$.content.split_mode: str 'text'
$.content.examples: list len=417
$.content.examples[0]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.content.examples[0].id: str 'retasy_train_000029_chunk_00'
$.content.examples[0].gold: str 'منشر'
$.content.examples[0].pred: str 'مننشششر'
$.content.examples[0].exact: bool False
$.content.examples[0].char_accuracy: float 0.25
$.content.examples[0].edit_distance: int 3
$.content.examples[0].gold_len: int 4
$.content.examples[0].pred_len: int 7
$.content.examples[1]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.content.examples[1].id: str 'retasy_train_000225_chunk_00'
$.content.examples[1].gold: str 'منشر'
$.content.examples[1].pred: str 'منششرر'
$.content.examples[1].exact: bool False
$.content.examples[1].char_accuracy: float 0.5
$.content.examples[1].edit_distance: int 2
$.content.examples[1].gold_len: int 4
$.content.examples[1].pred_len: int 6
$.content.examples[2]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.content.examples[2].id: str 'retasy_train_000355_chunk_00'
$.content.examples[2].gold: str 'منشر'
$.content.examples[2].pred: str 'منشششرر'
$.content.examples[2].exact: bool False
$.content.examples[2].char_accuracy: float 0.25
$.content.examples[2].edit_distance: int 3
$.content.examples[2].gold_len: int 4
```

### `data\analysis\ablations\modular_content_eval_lexicon_bp04.json`

```text
$: dict keys=['duration', 'transition', 'burst', 'content', 'content_reference_full_verse', 'weighted_scoring', 'duration_checkpoint', 'transition_checkpoint', 'ablation_flags']
$.duration: dict keys=['samples', 'route_counts', 'total_positions', 'correct_positions', 'accuracy', 'rule_summary', 'hybrid_support']
$.duration.samples: int 973
$.duration.route_counts: dict keys=['duration']
$.duration.route_counts.duration: int 973
$.duration.total_positions: int 1646
$.duration.correct_positions: int 1634
$.duration.accuracy: float 0.9927095990279465
$.duration.rule_summary: dict keys=['ghunnah', 'madd']
$.duration.rule_summary.ghunnah: dict keys=['total', 'correct', 'accuracy']
$.duration.rule_summary.ghunnah.total: int 364
$.duration.rule_summary.ghunnah.correct: int 358
$.duration.rule_summary.ghunnah.accuracy: float 0.9835164835164835
$.duration.rule_summary.madd: dict keys=['total', 'correct', 'accuracy']
$.duration.rule_summary.madd.total: int 1282
$.duration.rule_summary.madd.correct: int 1276
$.duration.rule_summary.madd.accuracy: float 0.9953198127925117
$.duration.hybrid_support: dict keys=['localized_available', 'localized_same_as_sequence', 'localized_same_rate', 'localized_supports_gold', 'localized_supports_gold_rate', 'localized_supports_sequence', 'localized_supports_sequence_rate', 'localized_disagrees_with_sequence', 'gold_supported_by_class', 'sequence_supported_by_class']
$.duration.hybrid_support.localized_available: int 1646
$.duration.hybrid_support.localized_same_as_sequence: int 1607
$.duration.hybrid_support.localized_same_rate: float 0.9763061968408262
$.duration.hybrid_support.localized_supports_gold: int 1600
$.duration.hybrid_support.localized_supports_gold_rate: float 0.9720534629404617
$.duration.hybrid_support.localized_supports_sequence: int 1607
$.duration.hybrid_support.localized_supports_sequence_rate: float 0.9763061968408262
$.duration.hybrid_support.localized_disagrees_with_sequence: int 39
$.transition: dict keys=['available', 'samples', 'accuracy', 'confusion_matrix', 'class_summary', 'hybrid_support']
$.transition.available: bool True
$.transition.samples: int 690
$.transition.accuracy: float 0.9101449275362319
$.transition.confusion_matrix: list len=3
$.transition.confusion_matrix[0]: list len=3
$.transition.confusion_matrix[0][0]: int 381
$.transition.confusion_matrix[0][1]: int 22
$.transition.confusion_matrix[0][2]: int 11
$.transition.confusion_matrix[1]: list len=3
$.transition.confusion_matrix[1][0]: int 21
$.transition.confusion_matrix[1][1]: int 205
$.transition.confusion_matrix[1][2]: int 1
$.transition.confusion_matrix[2]: list len=3
$.transition.confusion_matrix[2][0]: int 7
$.transition.confusion_matrix[2][1]: int 0
$.transition.confusion_matrix[2][2]: int 42
$.transition.class_summary: dict keys=['none', 'ikhfa', 'idgham']
$.transition.class_summary.none: dict keys=['total', 'correct', 'accuracy']
$.transition.class_summary.none.total: int 414
$.transition.class_summary.none.correct: int 381
$.transition.class_summary.none.accuracy: float 0.9202898550724637
$.transition.class_summary.ikhfa: dict keys=['total', 'correct', 'accuracy']
$.transition.class_summary.ikhfa.total: int 227
$.transition.class_summary.ikhfa.correct: int 205
$.transition.class_summary.ikhfa.accuracy: float 0.9030837004405287
$.transition.class_summary.idgham: dict keys=['total', 'correct', 'accuracy']
$.transition.class_summary.idgham.total: int 49
$.transition.class_summary.idgham.correct: int 42
$.transition.class_summary.idgham.accuracy: float 0.8571428571428571
$.transition.hybrid_support: dict keys=['localized_available', 'localized_same_as_whole_verse', 'localized_same_rate', 'localized_supports_gold', 'localized_supports_gold_rate', 'localized_supports_whole_verse', 'localized_supports_whole_verse_rate', 'localized_disagrees_with_whole_verse', 'gold_supported_by_class', 'whole_verse_supported_by_class']
$.transition.hybrid_support.localized_available: int 308
$.transition.hybrid_support.localized_same_as_whole_verse: int 268
$.transition.hybrid_support.localized_same_rate: float 0.8701298701298701
$.transition.hybrid_support.localized_supports_gold: int 112
$.transition.hybrid_support.localized_supports_gold_rate: float 0.36363636363636365
$.transition.hybrid_support.localized_supports_whole_verse: int 104
$.transition.hybrid_support.localized_supports_whole_verse_rate: float 0.33766233766233766
$.transition.hybrid_support.localized_disagrees_with_whole_verse: int 40
$.burst: dict keys=['available', 'samples', 'accuracy', 'confusion_matrix', 'class_summary', 'burst_threshold', 'decision_rule']
$.burst.available: bool True
$.burst.samples: int 1597
$.burst.accuracy: float 0.8753913587977458
$.burst.confusion_matrix: list len=2
$.burst.confusion_matrix[0]: list len=2
$.burst.confusion_matrix[0][0]: int 857
$.burst.confusion_matrix[0][1]: int 101
$.burst.confusion_matrix[1]: list len=2
$.burst.confusion_matrix[1][0]: int 98
$.burst.confusion_matrix[1][1]: int 541
$.burst.class_summary: dict keys=['none', 'qalqalah']
$.burst.class_summary.none: dict keys=['total', 'correct', 'accuracy']
$.burst.class_summary.none.total: int 958
$.burst.class_summary.none.correct: int 857
$.burst.class_summary.none.accuracy: float 0.894572025052192
$.burst.class_summary.qalqalah: dict keys=['total', 'correct', 'accuracy']
$.burst.class_summary.qalqalah.total: int 639
$.burst.class_summary.qalqalah.correct: int 541
$.burst.class_summary.qalqalah.accuracy: float 0.8466353677621283
$.burst.burst_threshold: float 0.47
$.burst.decision_rule: str 'qalqalah_probability_threshold'
$.content: dict keys=['available', 'mode', 'samples', 'split', 'split_mode', 'examples', 'worst_examples', 'decoder', 'exact_match', 'char_accuracy', 'edit_distance']
$.content.available: bool True
$.content.mode: str 'chunked'
$.content.samples: int 417
$.content.split: str 'val'
$.content.split_mode: str 'text'
$.content.examples: list len=417
$.content.examples[0]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.content.examples[0].id: str 'retasy_train_000029_chunk_00'
$.content.examples[0].gold: str 'منشر'
$.content.examples[0].pred: str 'زسخج'
$.content.examples[0].exact: bool False
$.content.examples[0].char_accuracy: float 0.0
$.content.examples[0].edit_distance: int 4
$.content.examples[0].gold_len: int 4
$.content.examples[0].pred_len: int 4
$.content.examples[1]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.content.examples[1].id: str 'retasy_train_000225_chunk_00'
$.content.examples[1].gold: str 'منشر'
$.content.examples[1].pred: str 'زسخج'
$.content.examples[1].exact: bool False
$.content.examples[1].char_accuracy: float 0.0
$.content.examples[1].edit_distance: int 4
$.content.examples[1].gold_len: int 4
$.content.examples[1].pred_len: int 4
$.content.examples[2]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.content.examples[2].id: str 'retasy_train_000355_chunk_00'
$.content.examples[2].gold: str 'منشر'
$.content.examples[2].pred: str 'زسخج'
$.content.examples[2].exact: bool False
$.content.examples[2].char_accuracy: float 0.0
$.content.examples[2].edit_distance: int 4
$.content.examples[2].gold_len: int 4
```

### `data\analysis\ablations\modular_content_reciter_split_burst047.json`

```text
$: dict keys=['duration', 'transition', 'burst', 'content', 'content_reference_full_verse', 'weighted_scoring', 'duration_checkpoint', 'transition_checkpoint', 'ablation_flags']
$.duration: dict keys=['samples', 'route_counts', 'total_positions', 'correct_positions', 'accuracy', 'rule_summary', 'hybrid_support']
$.duration.samples: int 973
$.duration.route_counts: dict keys=['duration']
$.duration.route_counts.duration: int 973
$.duration.total_positions: int 1646
$.duration.correct_positions: int 1634
$.duration.accuracy: float 0.9927095990279465
$.duration.rule_summary: dict keys=['ghunnah', 'madd']
$.duration.rule_summary.ghunnah: dict keys=['total', 'correct', 'accuracy']
$.duration.rule_summary.ghunnah.total: int 364
$.duration.rule_summary.ghunnah.correct: int 358
$.duration.rule_summary.ghunnah.accuracy: float 0.9835164835164835
$.duration.rule_summary.madd: dict keys=['total', 'correct', 'accuracy']
$.duration.rule_summary.madd.total: int 1282
$.duration.rule_summary.madd.correct: int 1276
$.duration.rule_summary.madd.accuracy: float 0.9953198127925117
$.duration.hybrid_support: dict keys=['localized_available', 'localized_same_as_sequence', 'localized_same_rate', 'localized_supports_gold', 'localized_supports_gold_rate', 'localized_supports_sequence', 'localized_supports_sequence_rate', 'localized_disagrees_with_sequence', 'gold_supported_by_class', 'sequence_supported_by_class']
$.duration.hybrid_support.localized_available: int 1646
$.duration.hybrid_support.localized_same_as_sequence: int 1607
$.duration.hybrid_support.localized_same_rate: float 0.9763061968408262
$.duration.hybrid_support.localized_supports_gold: int 1600
$.duration.hybrid_support.localized_supports_gold_rate: float 0.9720534629404617
$.duration.hybrid_support.localized_supports_sequence: int 1607
$.duration.hybrid_support.localized_supports_sequence_rate: float 0.9763061968408262
$.duration.hybrid_support.localized_disagrees_with_sequence: int 39
$.transition: dict keys=['available', 'samples', 'accuracy', 'confusion_matrix', 'class_summary', 'hybrid_support']
$.transition.available: bool True
$.transition.samples: int 690
$.transition.accuracy: float 0.9101449275362319
$.transition.confusion_matrix: list len=3
$.transition.confusion_matrix[0]: list len=3
$.transition.confusion_matrix[0][0]: int 381
$.transition.confusion_matrix[0][1]: int 22
$.transition.confusion_matrix[0][2]: int 11
$.transition.confusion_matrix[1]: list len=3
$.transition.confusion_matrix[1][0]: int 21
$.transition.confusion_matrix[1][1]: int 205
$.transition.confusion_matrix[1][2]: int 1
$.transition.confusion_matrix[2]: list len=3
$.transition.confusion_matrix[2][0]: int 7
$.transition.confusion_matrix[2][1]: int 0
$.transition.confusion_matrix[2][2]: int 42
$.transition.class_summary: dict keys=['none', 'ikhfa', 'idgham']
$.transition.class_summary.none: dict keys=['total', 'correct', 'accuracy']
$.transition.class_summary.none.total: int 414
$.transition.class_summary.none.correct: int 381
$.transition.class_summary.none.accuracy: float 0.9202898550724637
$.transition.class_summary.ikhfa: dict keys=['total', 'correct', 'accuracy']
$.transition.class_summary.ikhfa.total: int 227
$.transition.class_summary.ikhfa.correct: int 205
$.transition.class_summary.ikhfa.accuracy: float 0.9030837004405287
$.transition.class_summary.idgham: dict keys=['total', 'correct', 'accuracy']
$.transition.class_summary.idgham.total: int 49
$.transition.class_summary.idgham.correct: int 42
$.transition.class_summary.idgham.accuracy: float 0.8571428571428571
$.transition.hybrid_support: dict keys=['localized_available', 'localized_same_as_whole_verse', 'localized_same_rate', 'localized_supports_gold', 'localized_supports_gold_rate', 'localized_supports_whole_verse', 'localized_supports_whole_verse_rate', 'localized_disagrees_with_whole_verse', 'gold_supported_by_class', 'whole_verse_supported_by_class']
$.transition.hybrid_support.localized_available: int 308
$.transition.hybrid_support.localized_same_as_whole_verse: int 268
$.transition.hybrid_support.localized_same_rate: float 0.8701298701298701
$.transition.hybrid_support.localized_supports_gold: int 112
$.transition.hybrid_support.localized_supports_gold_rate: float 0.36363636363636365
$.transition.hybrid_support.localized_supports_whole_verse: int 104
$.transition.hybrid_support.localized_supports_whole_verse_rate: float 0.33766233766233766
$.transition.hybrid_support.localized_disagrees_with_whole_verse: int 40
$.burst: dict keys=['available', 'samples', 'accuracy', 'confusion_matrix', 'class_summary', 'burst_threshold', 'decision_rule']
$.burst.available: bool True
$.burst.samples: int 1597
$.burst.accuracy: float 0.8753913587977458
$.burst.confusion_matrix: list len=2
$.burst.confusion_matrix[0]: list len=2
$.burst.confusion_matrix[0][0]: int 857
$.burst.confusion_matrix[0][1]: int 101
$.burst.confusion_matrix[1]: list len=2
$.burst.confusion_matrix[1][0]: int 98
$.burst.confusion_matrix[1][1]: int 541
$.burst.class_summary: dict keys=['none', 'qalqalah']
$.burst.class_summary.none: dict keys=['total', 'correct', 'accuracy']
$.burst.class_summary.none.total: int 958
$.burst.class_summary.none.correct: int 857
$.burst.class_summary.none.accuracy: float 0.894572025052192
$.burst.class_summary.qalqalah: dict keys=['total', 'correct', 'accuracy']
$.burst.class_summary.qalqalah.total: int 639
$.burst.class_summary.qalqalah.correct: int 541
$.burst.class_summary.qalqalah.accuracy: float 0.8466353677621283
$.burst.burst_threshold: float 0.47
$.burst.decision_rule: str 'qalqalah_probability_threshold'
$.content: dict keys=['available', 'mode', 'samples', 'split', 'split_mode', 'examples', 'worst_examples', 'decoder', 'exact_match', 'char_accuracy', 'edit_distance']
$.content.available: bool True
$.content.mode: str 'chunked'
$.content.samples: int 389
$.content.split: str 'val'
$.content.split_mode: str 'reciter'
$.content.examples: list len=389
$.content.examples[0]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.content.examples[0].id: str 'retasy_train_005359_chunk_00'
$.content.examples[0].gold: str 'الرحمن'
$.content.examples[0].pred: str 'الرحمن'
$.content.examples[0].exact: bool True
$.content.examples[0].char_accuracy: float 1.0
$.content.examples[0].edit_distance: int 0
$.content.examples[0].gold_len: int 6
$.content.examples[0].pred_len: int 6
$.content.examples[1]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.content.examples[1].id: str 'retasy_train_005359_chunk_01'
$.content.examples[1].gold: str 'الرحيم'
$.content.examples[1].pred: str 'الرحيم'
$.content.examples[1].exact: bool True
$.content.examples[1].char_accuracy: float 1.0
$.content.examples[1].edit_distance: int 0
$.content.examples[1].gold_len: int 6
$.content.examples[1].pred_len: int 6
$.content.examples[2]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.content.examples[2].id: str 'retasy_train_005361_chunk_00'
$.content.examples[2].gold: str 'لااعبد'
$.content.examples[2].pred: str 'واياكنباس'
$.content.examples[2].exact: bool False
$.content.examples[2].char_accuracy: float 0.0
$.content.examples[2].edit_distance: int 6
$.content.examples[2].gold_len: int 6
```

### `data\analysis\ablations\modular_default_transition_argmax_burst047_with_examples.json`

```text
$: dict keys=['duration', 'transition', 'burst', 'content', 'content_reference_full_verse', 'weighted_scoring', 'duration_checkpoint', 'transition_checkpoint', 'ablation_flags']
$.duration: dict keys=['samples', 'route_counts', 'total_positions', 'correct_positions', 'accuracy', 'rule_summary', 'hybrid_support']
$.duration.samples: int 973
$.duration.route_counts: dict keys=['duration']
$.duration.route_counts.duration: int 973
$.duration.total_positions: int 1646
$.duration.correct_positions: int 1634
$.duration.accuracy: float 0.9927095990279465
$.duration.rule_summary: dict keys=['ghunnah', 'madd']
$.duration.rule_summary.ghunnah: dict keys=['total', 'correct', 'accuracy']
$.duration.rule_summary.ghunnah.total: int 364
$.duration.rule_summary.ghunnah.correct: int 358
$.duration.rule_summary.ghunnah.accuracy: float 0.9835164835164835
$.duration.rule_summary.madd: dict keys=['total', 'correct', 'accuracy']
$.duration.rule_summary.madd.total: int 1282
$.duration.rule_summary.madd.correct: int 1276
$.duration.rule_summary.madd.accuracy: float 0.9953198127925117
$.duration.hybrid_support: dict keys=['localized_available', 'localized_same_as_sequence', 'localized_same_rate', 'localized_supports_gold', 'localized_supports_gold_rate', 'localized_supports_sequence', 'localized_supports_sequence_rate', 'localized_disagrees_with_sequence', 'gold_supported_by_class', 'sequence_supported_by_class']
$.duration.hybrid_support.localized_available: int 1646
$.duration.hybrid_support.localized_same_as_sequence: int 1607
$.duration.hybrid_support.localized_same_rate: float 0.9763061968408262
$.duration.hybrid_support.localized_supports_gold: int 1600
$.duration.hybrid_support.localized_supports_gold_rate: float 0.9720534629404617
$.duration.hybrid_support.localized_supports_sequence: int 1607
$.duration.hybrid_support.localized_supports_sequence_rate: float 0.9763061968408262
$.duration.hybrid_support.localized_disagrees_with_sequence: int 39
$.transition: dict keys=['available', 'samples', 'accuracy', 'confusion_matrix', 'class_summary', 'hybrid_support']
$.transition.available: bool True
$.transition.samples: int 690
$.transition.accuracy: float 0.9101449275362319
$.transition.confusion_matrix: list len=3
$.transition.confusion_matrix[0]: list len=3
$.transition.confusion_matrix[0][0]: int 381
$.transition.confusion_matrix[0][1]: int 22
$.transition.confusion_matrix[0][2]: int 11
$.transition.confusion_matrix[1]: list len=3
$.transition.confusion_matrix[1][0]: int 21
$.transition.confusion_matrix[1][1]: int 205
$.transition.confusion_matrix[1][2]: int 1
$.transition.confusion_matrix[2]: list len=3
$.transition.confusion_matrix[2][0]: int 7
$.transition.confusion_matrix[2][1]: int 0
$.transition.confusion_matrix[2][2]: int 42
$.transition.class_summary: dict keys=['none', 'ikhfa', 'idgham']
$.transition.class_summary.none: dict keys=['total', 'correct', 'accuracy']
$.transition.class_summary.none.total: int 414
$.transition.class_summary.none.correct: int 381
$.transition.class_summary.none.accuracy: float 0.9202898550724637
$.transition.class_summary.ikhfa: dict keys=['total', 'correct', 'accuracy']
$.transition.class_summary.ikhfa.total: int 227
$.transition.class_summary.ikhfa.correct: int 205
$.transition.class_summary.ikhfa.accuracy: float 0.9030837004405287
$.transition.class_summary.idgham: dict keys=['total', 'correct', 'accuracy']
$.transition.class_summary.idgham.total: int 49
$.transition.class_summary.idgham.correct: int 42
$.transition.class_summary.idgham.accuracy: float 0.8571428571428571
$.transition.hybrid_support: dict keys=['localized_available', 'localized_same_as_whole_verse', 'localized_same_rate', 'localized_supports_gold', 'localized_supports_gold_rate', 'localized_supports_whole_verse', 'localized_supports_whole_verse_rate', 'localized_disagrees_with_whole_verse', 'gold_supported_by_class', 'whole_verse_supported_by_class']
$.transition.hybrid_support.localized_available: int 308
$.transition.hybrid_support.localized_same_as_whole_verse: int 268
$.transition.hybrid_support.localized_same_rate: float 0.8701298701298701
$.transition.hybrid_support.localized_supports_gold: int 112
$.transition.hybrid_support.localized_supports_gold_rate: float 0.36363636363636365
$.transition.hybrid_support.localized_supports_whole_verse: int 104
$.transition.hybrid_support.localized_supports_whole_verse_rate: float 0.33766233766233766
$.transition.hybrid_support.localized_disagrees_with_whole_verse: int 40
$.burst: dict keys=['available', 'samples', 'accuracy', 'confusion_matrix', 'class_summary', 'burst_threshold', 'decision_rule']
$.burst.available: bool True
$.burst.samples: int 1597
$.burst.accuracy: float 0.8753913587977458
$.burst.confusion_matrix: list len=2
$.burst.confusion_matrix[0]: list len=2
$.burst.confusion_matrix[0][0]: int 857
$.burst.confusion_matrix[0][1]: int 101
$.burst.confusion_matrix[1]: list len=2
$.burst.confusion_matrix[1][0]: int 98
$.burst.confusion_matrix[1][1]: int 541
$.burst.class_summary: dict keys=['none', 'qalqalah']
$.burst.class_summary.none: dict keys=['total', 'correct', 'accuracy']
$.burst.class_summary.none.total: int 958
$.burst.class_summary.none.correct: int 857
$.burst.class_summary.none.accuracy: float 0.894572025052192
$.burst.class_summary.qalqalah: dict keys=['total', 'correct', 'accuracy']
$.burst.class_summary.qalqalah.total: int 639
$.burst.class_summary.qalqalah.correct: int 541
$.burst.class_summary.qalqalah.accuracy: float 0.8466353677621283
$.burst.burst_threshold: float 0.47
$.burst.decision_rule: str 'qalqalah_probability_threshold'
$.content: dict keys=['available', 'mode', 'samples', 'split', 'split_mode', 'examples', 'worst_examples', 'decoder', 'exact_match', 'char_accuracy', 'edit_distance']
$.content.available: bool True
$.content.mode: str 'chunked'
$.content.samples: int 417
$.content.split: str 'val'
$.content.split_mode: str 'text'
$.content.examples: list len=417
$.content.examples[0]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.content.examples[0].id: str 'retasy_train_000029_chunk_00'
$.content.examples[0].gold: str 'منشر'
$.content.examples[0].pred: str 'منشر'
$.content.examples[0].exact: bool True
$.content.examples[0].char_accuracy: float 1.0
$.content.examples[0].edit_distance: int 0
$.content.examples[0].gold_len: int 4
$.content.examples[0].pred_len: int 4
$.content.examples[1]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.content.examples[1].id: str 'retasy_train_000225_chunk_00'
$.content.examples[1].gold: str 'منشر'
$.content.examples[1].pred: str 'منشر'
$.content.examples[1].exact: bool True
$.content.examples[1].char_accuracy: float 1.0
$.content.examples[1].edit_distance: int 0
$.content.examples[1].gold_len: int 4
$.content.examples[1].pred_len: int 4
$.content.examples[2]: dict keys=['id', 'gold', 'pred', 'exact', 'char_accuracy', 'edit_distance', 'gold_len', 'pred_len', 'len_delta']
$.content.examples[2].id: str 'retasy_train_000355_chunk_00'
$.content.examples[2].gold: str 'منشر'
$.content.examples[2].pred: str 'منشر'
$.content.examples[2].exact: bool True
$.content.examples[2].char_accuracy: float 1.0
$.content.examples[2].edit_distance: int 0
$.content.examples[2].gold_len: int 4
```

### `data\analysis\ablations\routing_profiles_retasy_all.json`

```text
$: dict keys=['dataset', 'split', 'threshold_config', 'profiles', 'samples', 'results']
$.dataset: str 'C:\\Users\\anis\\Desktop\\tajweed-modular-assessment\\data\\manifests\\learned_routing_dataset_v4_rule_aware_group_text.jsonl'
$.split: str 'all'
$.threshold_config: str 'C:\\Users\\anis\\Desktop\\tajweed-modular-assessment\\configs\\learned_router_v5_thresholds.yaml'
$.profiles: list len=2
$.profiles[0]: str 'trusted_retasy_calibrated'
$.profiles[1]: str 'weak_policy_tuned'
$.samples: int 1973
$.results: dict keys=['trusted_retasy_calibrated', 'weak_policy_tuned']
$.results.trusted_retasy_calibrated: dict keys=['samples', 'exact_agreement', 'macro_f1_vs_current', 'per_label', 'current_combo_counts', 'learned_combo_counts', 'disagreement_type_counts', 'examples']
$.results.trusted_retasy_calibrated.samples: int 1973
$.results.trusted_retasy_calibrated.exact_agreement: float 0.9751647237709072
$.results.trusted_retasy_calibrated.macro_f1_vs_current: float 0.9761648573960366
$.results.trusted_retasy_calibrated.per_label: dict keys=['use_duration', 'use_transition', 'use_burst']
$.results.trusted_retasy_calibrated.per_label.use_duration: dict keys=['accuracy', 'precision_vs_current', 'recall_vs_current', 'f1_vs_current', 'official_positive', 'learned_positive', 'tp_agree_on', 'fp_extra', 'fn_missed', 'tn_agree_off']
$.results.trusted_retasy_calibrated.per_label.use_transition: dict keys=['accuracy', 'precision_vs_current', 'recall_vs_current', 'f1_vs_current', 'official_positive', 'learned_positive', 'tp_agree_on', 'fp_extra', 'fn_missed', 'tn_agree_off']
$.results.trusted_retasy_calibrated.per_label.use_burst: dict keys=['accuracy', 'precision_vs_current', 'recall_vs_current', 'f1_vs_current', 'official_positive', 'learned_positive', 'tp_agree_on', 'fp_extra', 'fn_missed', 'tn_agree_off']
$.results.trusted_retasy_calibrated.current_combo_counts: dict keys=['use_duration', 'use_duration+use_transition', 'use_duration+use_burst', 'use_transition+use_burst', 'none', 'use_burst', 'use_transition']
$.results.trusted_retasy_calibrated.current_combo_counts.use_duration: int 808
$.results.trusted_retasy_calibrated.current_combo_counts.use_duration+use_transition: int 118
$.results.trusted_retasy_calibrated.current_combo_counts.use_duration+use_burst: int 47
$.results.trusted_retasy_calibrated.current_combo_counts.use_transition+use_burst: int 148
$.results.trusted_retasy_calibrated.current_combo_counts.none: int 398
$.results.trusted_retasy_calibrated.current_combo_counts.use_burst: int 444
$.results.trusted_retasy_calibrated.current_combo_counts.use_transition: int 10
$.results.trusted_retasy_calibrated.learned_combo_counts: dict keys=['use_duration', 'use_duration+use_transition', 'use_duration+use_burst', 'use_transition+use_burst', 'none', 'use_burst', 'use_transition', 'use_duration+use_transition+use_burst']
$.results.trusted_retasy_calibrated.learned_combo_counts.use_duration: int 808
$.results.trusted_retasy_calibrated.learned_combo_counts.use_duration+use_transition: int 118
$.results.trusted_retasy_calibrated.learned_combo_counts.use_duration+use_burst: int 47
$.results.trusted_retasy_calibrated.learned_combo_counts.use_transition+use_burst: int 148
$.results.trusted_retasy_calibrated.learned_combo_counts.none: int 371
$.results.trusted_retasy_calibrated.learned_combo_counts.use_burst: int 450
$.results.trusted_retasy_calibrated.learned_combo_counts.use_transition: int 10
$.results.trusted_retasy_calibrated.learned_combo_counts.use_duration+use_transition+use_burst: int 21
$.results.trusted_retasy_calibrated.disagreement_type_counts: dict keys=['extra_learned_module', 'missed_and_extra']
$.results.trusted_retasy_calibrated.disagreement_type_counts.extra_learned_module: int 48
$.results.trusted_retasy_calibrated.disagreement_type_counts.missed_and_extra: int 1
$.results.trusted_retasy_calibrated.examples: list len=40
$.results.trusted_retasy_calibrated.examples[0]: dict keys=['id', 'text', 'label_source', 'sources', 'current', 'learned', 'probabilities', 'missed', 'extra']
$.results.trusted_retasy_calibrated.examples[1]: dict keys=['id', 'text', 'label_source', 'sources', 'current', 'learned', 'probabilities', 'missed', 'extra']
$.results.trusted_retasy_calibrated.examples[2]: dict keys=['id', 'text', 'label_source', 'sources', 'current', 'learned', 'probabilities', 'missed', 'extra']
$.results.weak_policy_tuned: dict keys=['samples', 'exact_agreement', 'macro_f1_vs_current', 'per_label', 'current_combo_counts', 'learned_combo_counts', 'disagreement_type_counts', 'examples']
$.results.weak_policy_tuned.samples: int 1973
$.results.weak_policy_tuned.exact_agreement: float 0.9716168271667511
$.results.weak_policy_tuned.macro_f1_vs_current: float 0.970047455943552
$.results.weak_policy_tuned.per_label: dict keys=['use_duration', 'use_transition', 'use_burst']
$.results.weak_policy_tuned.per_label.use_duration: dict keys=['accuracy', 'precision_vs_current', 'recall_vs_current', 'f1_vs_current', 'official_positive', 'learned_positive', 'tp_agree_on', 'fp_extra', 'fn_missed', 'tn_agree_off']
$.results.weak_policy_tuned.per_label.use_transition: dict keys=['accuracy', 'precision_vs_current', 'recall_vs_current', 'f1_vs_current', 'official_positive', 'learned_positive', 'tp_agree_on', 'fp_extra', 'fn_missed', 'tn_agree_off']
$.results.weak_policy_tuned.per_label.use_burst: dict keys=['accuracy', 'precision_vs_current', 'recall_vs_current', 'f1_vs_current', 'official_positive', 'learned_positive', 'tp_agree_on', 'fp_extra', 'fn_missed', 'tn_agree_off']
$.results.weak_policy_tuned.current_combo_counts: dict keys=['use_duration', 'use_duration+use_transition', 'use_duration+use_burst', 'use_transition+use_burst', 'none', 'use_burst', 'use_transition']
$.results.weak_policy_tuned.current_combo_counts.use_duration: int 808
$.results.weak_policy_tuned.current_combo_counts.use_duration+use_transition: int 118
$.results.weak_policy_tuned.current_combo_counts.use_duration+use_burst: int 47
$.results.weak_policy_tuned.current_combo_counts.use_transition+use_burst: int 148
$.results.weak_policy_tuned.current_combo_counts.none: int 398
$.results.weak_policy_tuned.current_combo_counts.use_burst: int 444
$.results.weak_policy_tuned.current_combo_counts.use_transition: int 10
$.results.weak_policy_tuned.learned_combo_counts: dict keys=['use_duration', 'use_duration+use_transition', 'use_duration+use_burst', 'use_duration+use_transition+use_burst', 'use_transition+use_burst', 'none', 'use_burst', 'use_transition']
$.results.weak_policy_tuned.learned_combo_counts.use_duration: int 808
$.results.weak_policy_tuned.learned_combo_counts.use_duration+use_transition: int 118
$.results.weak_policy_tuned.learned_combo_counts.use_duration+use_burst: int 58
$.results.weak_policy_tuned.learned_combo_counts.use_duration+use_transition+use_burst: int 26
$.results.weak_policy_tuned.learned_combo_counts.use_transition+use_burst: int 144
$.results.weak_policy_tuned.learned_combo_counts.none: int 373
$.results.weak_policy_tuned.learned_combo_counts.use_burst: int 428
$.results.weak_policy_tuned.learned_combo_counts.use_transition: int 18
$.results.weak_policy_tuned.disagreement_type_counts: dict keys=['extra_learned_module', 'missed_official_module', 'missed_and_extra']
$.results.weak_policy_tuned.disagreement_type_counts.extra_learned_module: int 53
$.results.weak_policy_tuned.disagreement_type_counts.missed_official_module: int 2
$.results.weak_policy_tuned.disagreement_type_counts.missed_and_extra: int 1
$.results.weak_policy_tuned.examples: list len=40
$.results.weak_policy_tuned.examples[0]: dict keys=['id', 'text', 'label_source', 'sources', 'current', 'learned', 'probabilities', 'missed', 'extra']
$.results.weak_policy_tuned.examples[1]: dict keys=['id', 'text', 'label_source', 'sources', 'current', 'learned', 'probabilities', 'missed', 'extra']
$.results.weak_policy_tuned.examples[2]: dict keys=['id', 'text', 'label_source', 'sources', 'current', 'learned', 'probabilities', 'missed', 'extra']
```
