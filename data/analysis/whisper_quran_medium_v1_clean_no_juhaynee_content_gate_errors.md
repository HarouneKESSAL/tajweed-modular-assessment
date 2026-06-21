# Whisper content gate error report

This report analyzes Whisper ASR errors after Quran normalization and muqattaat normalization.

## Overall after muqattaat normalization

| metric | value |
|---|---:|
| samples | 407 |
| exact_after_rate | 0.7346 |
| avg_char_accuracy_after | 0.9803 |
| cer_after | 0.0205 |
| avg_gold_len | 27.7617 |
| avg_pred_len | 27.6265 |
| num_errors_after_muqattaat | 108 |
| num_near_misses_ge_095 | 61 |
| num_strong_near_misses_ge_098 | 6 |
| num_muqattaat_changed | 10 |

## Error type counts

| type | count |
|---|---:|
| single_edit_errors | 71 |
| two_edit_errors | 21 |
| short_predictions_len_delta_le_minus2 | 8 |
| long_predictions_len_delta_ge_2 | 6 |
| muqattaat_remaining_errors | 0 |

## By length

| bucket | samples | exact_after | char_acc | CER | avg_gold_len |
|---|---:|---:|---:|---:|---:|
| 001_010 | 15 | 1.000 | 1.000 | 0.000 | 4.93 |
| 011_020 | 142 | 0.739 | 0.972 | 0.028 | 15.80 |
| 021_040 | 161 | 0.795 | 0.989 | 0.012 | 29.48 |
| 041_060 | 89 | 0.573 | 0.975 | 0.026 | 47.58 |

## By reciter, worst CER first

| reciter | samples | exact_after | char_acc | CER |
|---|---:|---:|---:|---:|
| husary_mujawwad | 19 | 0.474 | 0.929 | 0.086 |
| minshawy_mujawwad | 12 | 0.333 | 0.936 | 0.068 |
| muhsin_al_qasim | 36 | 0.611 | 0.956 | 0.038 |
| warsh_husary | 5 | 0.600 | 0.975 | 0.027 |
| hussary.teacher | 31 | 0.871 | 0.972 | 0.021 |
| saood_ash_shuraym | 13 | 0.769 | 0.980 | 0.020 |
| ghamadi | 29 | 0.862 | 0.989 | 0.015 |
| warsh_yassin | 31 | 0.645 | 0.986 | 0.015 |
| banna | 31 | 0.548 | 0.988 | 0.013 |
| abdul_basit_murattal | 46 | 0.717 | 0.987 | 0.013 |
| abu_bakr_ash_shaatree | 56 | 0.821 | 0.988 | 0.012 |
| abdullah_basfar | 11 | 0.727 | 0.990 | 0.010 |
| ali_jaber | 20 | 0.850 | 0.988 | 0.009 |
| alafasy | 23 | 0.783 | 0.992 | 0.008 |
| ibrahim_akhdar | 33 | 0.909 | 0.996 | 0.004 |
| abdurrahmaan_as_sudais | 11 | 0.909 | 0.998 | 0.002 |

## Character-level clues

- top deleted gold chars: `[('ا', 30), ('و', 12), ('ن', 10), ('م', 9), ('ي', 8), ('ل', 8), ('ه', 6), ('ت', 4), ('د', 3), ('ش', 2), ('ر', 2), ('ء', 2), ('س', 1), ('ك', 1), ('ؤ', 1), ('ع', 1), ('ذ', 1), ('ج', 1), ('ق', 1), ('خ', 1), ('ص', 1), ('ض', 1)]`
- top inserted pred chars: `[('ا', 27), ('و', 15), ('ن', 13), ('ي', 12), ('ف', 9), ('ل', 7), ('غ', 5), ('ق', 5), ('ب', 4), ('ك', 4), ('د', 3), ('ط', 3), ('ر', 3), ('ه', 3), ('ذ', 2), ('س', 2), ('ئ', 2), ('م', 2), ('ء', 1), ('ج', 1), ('ظ', 1), ('ش', 1), ('ع', 1), ('ض', 1), ('ص', 1)]`
- top replaced gold chars: `[('ا', 9), ('ي', 8), ('ذ', 6), ('ه', 6), ('ل', 6), ('ن', 5), ('ر', 4), ('ف', 4), ('ض', 4), ('م', 3), ('و', 3), ('ق', 3), ('ء', 2), ('ظ', 2), ('ث', 2), ('ت', 2), ('ب', 1), ('د', 1), ('ج', 1), ('ك', 1), ('ئ', 1), ('س', 1), ('ص', 1), ('خ', 1)]`

## Muqattaat changed examples

### hf_quran_md_ayah_route_saood_ash_shuraym_002_001_000210
- gold: `الم`
- pred_raw: `الافلاميم`
- pred_after: `الم`
- exact_after: True

### hf_quran_md_ayah_route_saood_ash_shuraym_003_001_008790
- gold: `الم`
- pred_raw: `الافلاميم`
- pred_after: `الم`
- exact_after: True

### hf_quran_md_ayah_route_abdul_basit_murattal_026_001_087967
- gold: `طسم`
- pred_raw: `طاسيم`
- pred_after: `طسم`
- exact_after: True

### hf_quran_md_ayah_route_ali_jaber_028_001_097568
- gold: `طسم`
- pred_raw: `باسيم`
- pred_after: `طسم`
- exact_after: True

### hf_quran_md_ayah_route_ali_jaber_029_001_100208
- gold: `الم`
- pred_raw: `الفلاميم`
- pred_after: `الم`
- exact_after: True

### hf_quran_md_ayah_route_ali_jaber_030_001_102278
- gold: `الم`
- pred_raw: `الفلاميم`
- pred_after: `الم`
- exact_after: True

### hf_quran_md_ayah_route_ali_jaber_031_001_104078
- gold: `الم`
- pred_raw: `الفلاميم`
- pred_after: `الم`
- exact_after: True

### hf_quran_md_ayah_route_ali_jaber_032_001_105098
- gold: `الم`
- pred_raw: `الفلاميم`
- pred_after: `الم`
- exact_after: True

### hf_quran_md_ayah_route_hussary.teacher_036_001_111159
- gold: `يس`
- pred_raw: `يا سين`
- pred_after: `يس`
- exact_after: True

### hf_quran_md_ayah_route_alafasy_042_002_128200
- gold: `عسق`
- pred_raw: `عين سينقاف`
- pred_after: `عسق`
- exact_after: True


## Strong near misses, char accuracy >= 0.98

### hf_quran_md_ayah_route_banna_017_056_062525
- reciter: `banna`
- gold: `قل ادعوا الذين زعمتم من دونه فلا يملكون كشف الضر عنكم ولا تحويلا`
- pred: `قل ادعو الذين زعمتم من دونه فلا يملكون كشف الضر عنكم ولا تحويلا`
- char_accuracy: 0.9808
- edit_distance: 1
- len_delta: -1

### hf_quran_md_ayah_route_banna_019_004_067595
- reciter: `banna`
- gold: `قال رب اني وهن العظم مني واشتعل الراس شيبا ولم اكن بدعائك رب شقيا`
- pred: `قال رب اني وهني العظم مني واشتعل الراس شيبا ولم اكن بدعائك رب شقيا`
- char_accuracy: 0.9808
- edit_distance: 1
- len_delta: 1

### hf_quran_md_ayah_route_banna_019_048_068915
- reciter: `banna`
- gold: `واعتزلكم وما تدعون من دون الله وادعو ربي عسي الا اكون بدعاء ربي شقيا`
- pred: `واعتزلكم وما تدعون من دون الله وادعوا ربي عسي الا اكون بدعاء ربي شقيا`
- char_accuracy: 0.9818
- edit_distance: 1
- len_delta: 1

### hf_quran_md_ayah_route_abdul_basit_murattal_021_056_076147
- reciter: `abdul_basit_murattal`
- gold: `قال بل ربكم رب السماوات والارض الذي فطرهن وانا علي ذلكم من الشاهدين`
- pred: `قال بل ربكم رب السماوات والارض الذي فطرهن وانا علي ذلك من الشاهدين`
- char_accuracy: 0.9818
- edit_distance: 1
- len_delta: -1

### hf_quran_md_ayah_route_warsh_yassin_022_022_078486
- reciter: `warsh_yassin`
- gold: `كلما ارادوا ان يخرجوا منها من غم اعيدوا فيها وذوقوا عذاب الحريق`
- pred: `كلما ارادوا ان يخرجوا منها من غم عيدوا فيها وذوقوا عذاب الحريق`
- char_accuracy: 0.9808
- edit_distance: 1
- len_delta: -1

### hf_quran_md_ayah_route_abdul_basit_murattal_026_041_089167
- reciter: `abdul_basit_murattal`
- gold: `فلما جاء السحره قالوا لفرعون ائن لنا لاجرا ان كنا نحن الغالبين`
- pred: `فلما جاء السحره قالوا لفرعون اان لنا لاجرا ان كنا نحن الغالبين`
- char_accuracy: 0.9804
- edit_distance: 1
- len_delta: 0


## Worst examples

### hf_quran_md_ayah_route_hussary.teacher_037_152_118179
- reciter: `hussary.teacher`
- gold: `ولد الله وانهم لكاذبون`
- pred: `من افكهم ليقولون ولد الله وانهم لكاذبون`
- char_accuracy: 0.2632
- edit_distance: 14
- len_delta: 14

### hf_quran_md_ayah_route_husary_mujawwad_012_099_050824
- reciter: `husary_mujawwad`
- gold: `فلما دخلوا علي يوسف اوي اليه ابويه وقال ادخلوا مصر ان شاء الله امنين`
- pred: `فلما دخلوا علي يوسف اواء له ابوه`
- char_accuracy: 0.4364
- edit_distance: 31
- len_delta: -29

### hf_quran_md_ayah_route_husary_mujawwad_012_089_050524
- reciter: `husary_mujawwad`
- gold: `قال هل علمتم ما فعلتم بيوسف واخيه اذ انتم جاهلون`
- pred: `قال هل علمتم ما فعلتم بيوسف واخي`
- char_accuracy: 0.6667
- edit_distance: 13
- len_delta: -13

### hf_quran_md_ayah_route_minshawy_mujawwad_006_041_024872
- reciter: `minshawy_mujawwad`
- gold: `بل اياه تدعون فيكشف ما تدعون اليه ان شاء وتنسون ما تشركون`
- pred: `بل اياه تدعون فيكشف ما تدعون اليه ان شاء`
- char_accuracy: 0.6957
- edit_distance: 14
- len_delta: -14

### hf_quran_md_ayah_route_muhsin_al_qasim_109_006_186376
- reciter: `muhsin_al_qasim`
- gold: `لكم دينكم ولي دين`
- pred: `اوانكم دينكم ولي دين`
- char_accuracy: 0.7143
- edit_distance: 4
- len_delta: 3

### hf_quran_md_ayah_route_minshawy_mujawwad_007_139_032762
- reciter: `minshawy_mujawwad`
- gold: `ان هؤلاء متبر ما هم فيه وباطل ما كانوا يعملون`
- pred: `ان انا انا متبغ ما هم فيذ وباطل ما كانوا يعملون`
- char_accuracy: 0.8056
- edit_distance: 7
- len_delta: 1

### hf_quran_md_ayah_route_ali_jaber_031_003_104138
- reciter: `ali_jaber`
- gold: `هدي ورحمه للمحسنين`
- pred: `عودا ورحمه للمحسنين`
- char_accuracy: 0.8125
- edit_distance: 3
- len_delta: 1

### hf_quran_md_ayah_route_muhsin_al_qasim_096_007_183376
- reciter: `muhsin_al_qasim`
- gold: `ان راه استغني`
- pred: `قراه استغني`
- char_accuracy: 0.8182
- edit_distance: 2
- len_delta: -1

### hf_quran_md_ayah_route_muhsin_al_qasim_096_018_183706
- reciter: `muhsin_al_qasim`
- gold: `سندع الزبانيه`
- pred: `وسندعو الزبانيه`
- char_accuracy: 0.8333
- edit_distance: 2
- len_delta: 2

### hf_quran_md_ayah_route_muhsin_al_qasim_091_004_181396
- reciter: `muhsin_al_qasim`
- gold: `والليل اذا يغشاها`
- pred: `الليل اذا يغشانها`
- char_accuracy: 0.8667
- edit_distance: 2
- len_delta: 0

### hf_quran_md_ayah_route_abdul_basit_murattal_026_080_090337
- reciter: `abdul_basit_murattal`
- gold: `واذا مرضت فهو يشفين`
- pred: `واذا مرت فهو يشكين`
- char_accuracy: 0.8750
- edit_distance: 2
- len_delta: -1

### hf_quran_md_ayah_route_husary_mujawwad_012_079_050224
- reciter: `husary_mujawwad`
- gold: `قال معاذ الله ان ناخذ الا من وجدنا متاعنا عنده انا اذا لظالمون`
- pred: `قال معاذ الله ان ناخذ الا من وجدنا متاعنا عنده انا اذا لغبنا`
- char_accuracy: 0.8800
- edit_distance: 6
- len_delta: -2

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_080_034_173745
- reciter: `abu_bakr_ash_shaatree`
- gold: `يوم يفر المرء من اخيه`
- pred: `يوم يفروا المرء من اخيه`
- char_accuracy: 0.8824
- edit_distance: 2
- len_delta: 2

### hf_quran_md_ayah_route_muhsin_al_qasim_107_002_185956
- reciter: `muhsin_al_qasim`
- gold: `فذلك الذي يدع اليتيم`
- pred: `وذلك الذي يدعي اليتيم`
- char_accuracy: 0.8824
- edit_distance: 2
- len_delta: 1

### hf_quran_md_ayah_route_ghamadi_053_019_144072
- reciter: `ghamadi`
- gold: `افرايتم اللات والعزي`
- pred: `افرايتم الا تول عزي`
- char_accuracy: 0.8889
- edit_distance: 2
- len_delta: -2

### hf_quran_md_ayah_route_ghamadi_061_004_154992
- reciter: `ghamadi`
- gold: `ان الله يحب الذين يقاتلون في سبيله صفا كانهم بنيان مرصوص`
- pred: `ان الله يحب الذين يقاتلون في سبيله صفا صفا كانهم بنيان مقصوس`
- char_accuracy: 0.8913
- edit_distance: 5
- len_delta: 3

### hf_quran_md_ayah_route_saood_ash_shuraym_002_018_000720
- reciter: `saood_ash_shuraym`
- gold: `صم بكم عمي فهم لا يرجعون`
- pred: `صم بك منعم فهم لا يرجعون`
- char_accuracy: 0.8947
- edit_distance: 2
- len_delta: 0

### hf_quran_md_ayah_route_saood_ash_shuraym_002_042_001440
- reciter: `saood_ash_shuraym`
- gold: `ولا تلبسوا الحق بالباطل وتكتموا الحق وانتم تعلمون`
- pred: `ولا تلبس الحق بالباطل وتكتم الحق وانتم تعلمون`
- char_accuracy: 0.9048
- edit_distance: 4
- len_delta: -4

### hf_quran_md_ayah_route_ibrahim_akhdar_053_006_143681
- reciter: `ibrahim_akhdar`
- gold: `ذو مره فاستوي`
- pred: `ذو مروه فاستوي`
- char_accuracy: 0.9091
- edit_distance: 1
- len_delta: 1

### hf_quran_md_ayah_route_muhsin_al_qasim_092_004_181846
- reciter: `muhsin_al_qasim`
- gold: `ان سعيكم لشتي`
- pred: `ان سعيكم لشتا`
- char_accuracy: 0.9091
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_muhsin_al_qasim_100_005_184516
- reciter: `muhsin_al_qasim`
- gold: `فوسطن به جمعا`
- pred: `اوسطن به جمعا`
- char_accuracy: 0.9091
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_077_033_169635
- reciter: `abu_bakr_ash_shaatree`
- gold: `كانه جمالت صفر`
- pred: `كانه جماله صفر`
- char_accuracy: 0.9167
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_muhsin_al_qasim_096_012_183526
- reciter: `muhsin_al_qasim`
- gold: `او امر بالتقوي`
- pred: `وامر بالتقوي`
- char_accuracy: 0.9167
- edit_distance: 1
- len_delta: -1

### hf_quran_md_ayah_route_muhsin_al_qasim_097_004_183856
- reciter: `muhsin_al_qasim`
- gold: `تنزل الملائكه والروح فيها باذن ربهم من كل امر`
- pred: `نزلوا الملائكه والروح فيها باذن ربهم من كل امر`
- char_accuracy: 0.9189
- edit_distance: 3
- len_delta: 1

### hf_quran_md_ayah_route_warsh_husary_077_001_168674
- reciter: `warsh_husary`
- gold: `والمرسلات عرفا`
- pred: `والموسلات عرفا`
- char_accuracy: 0.9231
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_muhsin_al_qasim_092_010_182026
- reciter: `muhsin_al_qasim`
- gold: `فسنيسره للعسري`
- pred: `فسنيسره للعسرا`
- char_accuracy: 0.9231
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_muhsin_al_qasim_101_010_184996
- reciter: `muhsin_al_qasim`
- gold: `وما ادراك ما هيه`
- pred: `وما ادراك ما هي`
- char_accuracy: 0.9231
- edit_distance: 1
- len_delta: -1

### hf_quran_md_ayah_route_warsh_yassin_023_082_082626
- reciter: `warsh_yassin`
- gold: `قالوا ااذا متنا وكنا ترابا وعظاما اانا لمبعوثون`
- pred: `قالوا ايذا متنا وكنا ترابا وعظام انا لمبعوثون`
- char_accuracy: 0.9250
- edit_distance: 3
- len_delta: -2

### hf_quran_md_ayah_route_ghamadi_055_039_148182
- reciter: `ghamadi`
- gold: `فيومئذ لا يسال عن ذنبه انس ولا جان`
- pred: `فيومئذ لا يسال عن دان به انس ولا جان`
- char_accuracy: 0.9259
- edit_distance: 2
- len_delta: 1

### hf_quran_md_ayah_route_husary_mujawwad_013_012_051544
- reciter: `husary_mujawwad`
- gold: `هو الذي يريكم البرق خوفا وطمعا وينشئ السحاب الثقال`
- pred: `هو الذي يريكم البرق خوفا وطمعا وينشئ السحاب الثقافين`
- char_accuracy: 0.9286
- edit_distance: 3
- len_delta: 2

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_079_043_172635
- reciter: `abu_bakr_ash_shaatree`
- gold: `فيم انت من ذكراها`
- pred: `فيما انت من ذكراها`
- char_accuracy: 0.9286
- edit_distance: 1
- len_delta: 1

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_084_002_176565
- reciter: `abu_bakr_ash_shaatree`
- gold: `واذنت لربها وحقت`
- pred: `واذلت لربها وحقت`
- char_accuracy: 0.9286
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_muhsin_al_qasim_102_004_185146
- reciter: `muhsin_al_qasim`
- gold: `ثم كلا سوف تعلمون`
- pred: `امكلا سوف تعلمون`
- char_accuracy: 0.9286
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_abdul_basit_murattal_027_021_095377
- reciter: `abdul_basit_murattal`
- gold: `لاعذبنه عذابا شديدا او لاذبحنه او لياتيني بسلطان مبين`
- pred: `لاعذبنه عذابا شديدا اولا اذبحنه اولي اتيني بشرطان مبين`
- char_accuracy: 0.9333
- edit_distance: 3
- len_delta: 1

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_087_017_178935
- reciter: `abu_bakr_ash_shaatree`
- gold: `والاخره خير وابقي`
- pred: `والافره خير وابقي`
- char_accuracy: 0.9333
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_muhsin_al_qasim_096_013_183556
- reciter: `muhsin_al_qasim`
- gold: `ارايت ان كذب وتولي`
- pred: `فارايت ان كذب وتولي`
- char_accuracy: 0.9333
- edit_distance: 1
- len_delta: 1

### hf_quran_md_ayah_route_muhsin_al_qasim_105_005_185776
- reciter: `muhsin_al_qasim`
- gold: `فجعلهم كعصف ماكول`
- pred: `فجعلهم كعصف ماكل`
- char_accuracy: 0.9333
- edit_distance: 1
- len_delta: -1

### hf_quran_md_ayah_route_minshawy_mujawwad_007_015_029042
- reciter: `minshawy_mujawwad`
- gold: `قال انك من المنظرين`
- pred: `قال انك من النظرين`
- char_accuracy: 0.9375
- edit_distance: 1
- len_delta: -1

### hf_quran_md_ayah_route_hussary.teacher_035_021_110409
- reciter: `hussary.teacher`
- gold: `ولا الظل ولا الحرور`
- pred: `ولا الظل ولا الحضور`
- char_accuracy: 0.9375
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_minshawy_mujawwad_008_008_035012
- reciter: `minshawy_mujawwad`
- gold: `ليحق الحق ويبطل الباطل ولو كره المجرمون`
- pred: `ليحق الحق ويلطل الباطل ولو كري المجرمون`
- char_accuracy: 0.9394
- edit_distance: 2
- len_delta: 0

### hf_quran_md_ayah_route_abdul_basit_murattal_026_087_090547
- reciter: `abdul_basit_murattal`
- gold: `ولا تخزني يوم يبعثون`
- pred: `ولا تخزلي يوم يبعثون`
- char_accuracy: 0.9412
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_079_039_172515
- reciter: `abu_bakr_ash_shaatree`
- gold: `فان الجحيم هي الماوي`
- pred: `فان الجحيم هي المغوي`
- char_accuracy: 0.9412
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_083_016_175905
- reciter: `abu_bakr_ash_shaatree`
- gold: `ثم انهم لصالو الجحيم`
- pred: `ثم انهم لصال الجحيم`
- char_accuracy: 0.9412
- edit_distance: 1
- len_delta: -1

### hf_quran_md_ayah_route_minshawy_mujawwad_007_080_030992
- reciter: `minshawy_mujawwad`
- gold: `ولوطا اذ قال لقومه اتاتون الفاحشه ما سبقكم بها من احد من العالمين`
- pred: `ولوطا ان قال لقوم ذي اتاتون الفاحشه ما سبقكم بها من احد من العالمين`
- char_accuracy: 0.9434
- edit_distance: 3
- len_delta: 1

### hf_quran_md_ayah_route_saood_ash_shuraym_002_278_008520
- reciter: `saood_ash_shuraym`
- gold: `يا ايها الذين امنوا اتقوا الله وذروا ما بقي من الربا ان كنتم مؤمنين`
- pred: `يا ايها الذين امنوا اتقوا الله وادروا ما بقي من الربي ان كنتم مؤمنين`
- char_accuracy: 0.9444
- edit_distance: 3
- len_delta: 1

### hf_quran_md_ayah_route_abdul_basit_murattal_026_203_094027
- reciter: `abdul_basit_murattal`
- gold: `فيقولوا هل نحن منظرون`
- pred: `فيقولوا هل نحن موظرون`
- char_accuracy: 0.9444
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_warsh_yassin_021_058_076206
- reciter: `warsh_yassin`
- gold: `فجعلهم جذاذا الا كبيرا لهم لعلهم اليه يرجعون`
- pred: `فجعلهم جذاب الا كبيرا لهم لعلهم اليه يرجعون`
- char_accuracy: 0.9459
- edit_distance: 2
- len_delta: -1

### hf_quran_md_ayah_route_husary_mujawwad_015_045_055384
- reciter: `husary_mujawwad`
- gold: `ان المتقين في جنات وعيون`
- pred: `ان المتكين في جنات وعيون`
- char_accuracy: 0.9500
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_warsh_yassin_020_107_073626
- reciter: `warsh_yassin`
- gold: `لا تري فيها عوجا ولا امتا`
- pred: `لا تري فيها عوجا ولا امتي`
- char_accuracy: 0.9500
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_hussary.teacher_037_069_115689
- reciter: `hussary.teacher`
- gold: `انهم الفوا اباءهم ضالين`
- pred: `انهم الف واباءهم ضالين`
- char_accuracy: 0.9500
- edit_distance: 1
- len_delta: -1

### hf_quran_md_ayah_route_alafasy_038_037_120190
- reciter: `alafasy`
- gold: `والشياطين كل بناء وغواص`
- pred: `والشياطين كل بنا وغواص`
- char_accuracy: 0.9500
- edit_distance: 1
- len_delta: -1

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_077_031_169575
- reciter: `abu_bakr_ash_shaatree`
- gold: `لا ظليل ولا يغني من اللهب`
- pred: `لا غليل ولا يغني من اللهب`
- char_accuracy: 0.9500
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_090_004_180795
- reciter: `abu_bakr_ash_shaatree`
- gold: `لقد خلقنا الانسان في كبد`
- pred: `فقد خلقنا الانسان في كبد`
- char_accuracy: 0.9500
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_husary_mujawwad_015_074_056254
- reciter: `husary_mujawwad`
- gold: `فجعلنا عاليها سافلها وامطرنا عليهم حجاره من سجيل`
- pred: `فجعلنا عليها سافلها وانطرنا عليهم حجاره من سجيل`
- char_accuracy: 0.9512
- edit_distance: 2
- len_delta: -1

### hf_quran_md_ayah_route_warsh_yassin_025_009_085896
- reciter: `warsh_yassin`
- gold: `انظر كيف ضربوا لك الامثال فضلوا فلا يستطيعون سبيلا`
- pred: `انظر كيف ضربوا لك لمثال فضلوا فلا يستطيعون سبيلا`
- char_accuracy: 0.9524
- edit_distance: 2
- len_delta: -2

### hf_quran_md_ayah_route_abdul_basit_murattal_026_224_094657
- reciter: `abdul_basit_murattal`
- gold: `والشعراء يتبعهم الغاوون`
- pred: `والشعراء يتبعهم الغاغون`
- char_accuracy: 0.9524
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_warsh_husary_076_025_168464
- reciter: `warsh_husary`
- gold: `واذكر اسم ربك بكره واصيلا`
- pred: `والكر اسم ربك بكره واصيلا`
- char_accuracy: 0.9524
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_abdul_basit_murattal_021_089_077137
- reciter: `abdul_basit_murattal`
- gold: `وزكريا اذ نادي ربه رب لا تذرني فردا وانت خير الوارثين`
- pred: `وزكريا اب نادي ربه رب لا تذرني فردا وانت خير الوارفين`
- char_accuracy: 0.9535
- edit_distance: 2
- len_delta: 0

### hf_quran_md_ayah_route_alafasy_038_040_120280
- reciter: `alafasy`
- gold: `وان له عندنا لزلفي وحسن ماب`
- pred: `وان له عندنا لزلفا وحسن ماب`
- char_accuracy: 0.9545
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_husary_mujawwad_015_070_056134
- reciter: `husary_mujawwad`
- gold: `قالوا اولم ننهك عن العالمين`
- pred: `قالوا اولا ننهك عن العالمين`
- char_accuracy: 0.9565
- edit_distance: 1
- len_delta: 0
