# Whisper content gate error report

This report analyzes Whisper ASR errors after Quran normalization and muqattaat normalization.

## Overall after muqattaat normalization

| metric | value |
|---|---:|
| samples | 407 |
| exact_after_rate | 0.7396 |
| avg_char_accuracy_after | 0.9817 |
| cer_after | 0.0189 |
| avg_gold_len | 27.7617 |
| avg_pred_len | 27.6511 |
| num_errors_after_muqattaat | 106 |
| num_near_misses_ge_095 | 57 |
| num_strong_near_misses_ge_098 | 5 |
| num_muqattaat_changed | 8 |

## Error type counts

| type | count |
|---|---:|
| single_edit_errors | 72 |
| two_edit_errors | 21 |
| short_predictions_len_delta_le_minus2 | 6 |
| long_predictions_len_delta_ge_2 | 5 |
| muqattaat_remaining_errors | 0 |

## By length

| bucket | samples | exact_after | char_acc | CER | avg_gold_len |
|---|---:|---:|---:|---:|---:|
| 001_010 | 15 | 1.000 | 1.000 | 0.000 | 4.93 |
| 011_020 | 142 | 0.732 | 0.975 | 0.025 | 15.80 |
| 021_040 | 161 | 0.801 | 0.989 | 0.012 | 29.48 |
| 041_060 | 89 | 0.596 | 0.977 | 0.024 | 47.58 |

## By reciter, worst CER first

| reciter | samples | exact_after | char_acc | CER |
|---|---:|---:|---:|---:|
| husary_mujawwad | 19 | 0.474 | 0.937 | 0.076 |
| minshawy_mujawwad | 12 | 0.167 | 0.941 | 0.063 |
| warsh_husary | 5 | 0.400 | 0.951 | 0.054 |
| muhsin_al_qasim | 36 | 0.639 | 0.964 | 0.031 |
| hussary.teacher | 31 | 0.871 | 0.972 | 0.021 |
| saood_ash_shuraym | 13 | 0.769 | 0.984 | 0.018 |
| warsh_yassin | 31 | 0.645 | 0.984 | 0.017 |
| ghamadi | 29 | 0.828 | 0.989 | 0.014 |
| abu_bakr_ash_shaatree | 56 | 0.786 | 0.986 | 0.014 |
| abdul_basit_murattal | 46 | 0.783 | 0.989 | 0.010 |
| alafasy | 23 | 0.739 | 0.991 | 0.010 |
| abdullah_basfar | 11 | 0.636 | 0.988 | 0.010 |
| banna | 31 | 0.742 | 0.993 | 0.008 |
| ali_jaber | 20 | 0.900 | 0.996 | 0.005 |
| abdurrahmaan_as_sudais | 11 | 0.818 | 0.996 | 0.005 |
| ibrahim_akhdar | 33 | 0.909 | 0.996 | 0.004 |

## Character-level clues

- top deleted gold chars: `[('ا', 23), ('و', 11), ('ن', 8), ('م', 7), ('ي', 5), ('ه', 5), ('ت', 4), ('ل', 4), ('ر', 2), ('س', 1), ('ش', 1), ('ك', 1), ('ذ', 1), ('ج', 1), ('ظ', 1), ('ض', 1)]`
- top inserted pred chars: `[('ا', 22), ('و', 17), ('ي', 12), ('ن', 9), ('ل', 8), ('غ', 6), ('ب', 6), ('ف', 6), ('د', 4), ('ذ', 4), ('ك', 3), ('ر', 3), ('م', 3), ('ع', 3), ('ه', 3), ('س', 2), ('ق', 2), ('ظ', 2), ('ث', 1), ('ئ', 1), ('ء', 1), ('ط', 1), ('ص', 1)]`
- top replaced gold chars: `[('ا', 13), ('ه', 8), ('م', 7), ('ل', 7), ('ن', 7), ('ذ', 6), ('ف', 5), ('ر', 5), ('ي', 5), ('ق', 4), ('و', 4), ('ء', 2), ('ظ', 2), ('د', 2), ('خ', 2), ('ض', 2), ('ب', 1), ('ص', 1), ('ش', 1), ('ج', 1), ('ك', 1), ('ئ', 1), ('ت', 1), ('ث', 1)]`

## Muqattaat changed examples

### hf_quran_md_ayah_route_saood_ash_shuraym_002_001_000210
- gold: `الم`
- pred_raw: `الافلاميم`
- pred_after: `الم`
- exact_after: True

### hf_quran_md_ayah_route_saood_ash_shuraym_003_001_008790
- gold: `الم`
- pred_raw: `الميم`
- pred_after: `الم`
- exact_after: True

### hf_quran_md_ayah_route_abdul_basit_murattal_026_001_087967
- gold: `طسم`
- pred_raw: `طاسيميم`
- pred_after: `طسم`
- exact_after: True

### hf_quran_md_ayah_route_ali_jaber_028_001_097568
- gold: `طسم`
- pred_raw: `باسيم`
- pred_after: `طسم`
- exact_after: True

### hf_quran_md_ayah_route_ali_jaber_030_001_102278
- gold: `الم`
- pred_raw: `الميم`
- pred_after: `الم`
- exact_after: True

### hf_quran_md_ayah_route_ali_jaber_032_001_105098
- gold: `الم`
- pred_raw: `الميم`
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

### hf_quran_md_ayah_route_husary_mujawwad_013_021_051814
- reciter: `husary_mujawwad`
- gold: `والذين يصلون ما امر الله به ان يوصل ويخشون ربهم ويخافون سوء الحساب`
- pred: `والذين يصلون ما امر الله به ان يصل ويخشون ربهم ويخافون سوء الحساب`
- char_accuracy: 0.9815
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

### hf_quran_md_ayah_route_warsh_yassin_022_022_078486
- reciter: `warsh_yassin`
- gold: `كلما ارادوا ان يخرجوا منها من غم اعيدوا فيها وذوقوا عذاب الحريق`
- pred: `كل ما ارادوا ان يخرجوا منها من غم عيدوا فيها وذوقوا عذاب الحريق`
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
- pred: `فلما دخلوا علي يوسف اوي اليه ابو يوسف`
- char_accuracy: 0.5091
- edit_distance: 27
- len_delta: -25

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

### hf_quran_md_ayah_route_muhsin_al_qasim_096_007_183376
- reciter: `muhsin_al_qasim`
- gold: `ان راه استغني`
- pred: `ارواه استغني`
- char_accuracy: 0.8182
- edit_distance: 2
- len_delta: 0

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

### hf_quran_md_ayah_route_warsh_yassin_023_009_080436
- reciter: `warsh_yassin`
- gold: `والذين هم علي صلواتهم يحافظون`
- pred: `والذين هم علي صلواته ابن يحافظون`
- char_accuracy: 0.8800
- edit_distance: 3
- len_delta: 2

### hf_quran_md_ayah_route_abdul_basit_murattal_026_199_093907
- reciter: `abdul_basit_murattal`
- gold: `فقراه عليهم ما كانوا به مؤمنين`
- pred: `فقر ابو عليه ما كانوا به مؤمنين`
- char_accuracy: 0.8800
- edit_distance: 3
- len_delta: 0

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
- pred: `وذلك الذي يدعو اليتيم`
- char_accuracy: 0.8824
- edit_distance: 2
- len_delta: 1

### hf_quran_md_ayah_route_husary_mujawwad_012_079_050224
- reciter: `husary_mujawwad`
- gold: `قال معاذ الله ان ناخذ الا من وجدنا متاعنا عنده انا اذا لظالمون`
- pred: `قال معاذ الله ان ناخذ الا من وجدنا متاعنا عنده انا اذا لغبر معا`
- char_accuracy: 0.9000
- edit_distance: 5
- len_delta: 0

### hf_quran_md_ayah_route_saood_ash_shuraym_002_042_001440
- reciter: `saood_ash_shuraym`
- gold: `ولا تلبسوا الحق بالباطل وتكتموا الحق وانتم تعلمون`
- pred: `ولا تلبس الحق بالباطل وتكتم الحق وانتم تعلمون`
- char_accuracy: 0.9048
- edit_distance: 4
- len_delta: -4

### hf_quran_md_ayah_route_warsh_husary_076_025_168464
- reciter: `warsh_husary`
- gold: `واذكر اسم ربك بكره واصيلا`
- pred: `والكرس مربك بكره واصيلا`
- char_accuracy: 0.9048
- edit_distance: 2
- len_delta: -1

### hf_quran_md_ayah_route_ibrahim_akhdar_053_006_143681
- reciter: `ibrahim_akhdar`
- gold: `ذو مره فاستوي`
- pred: `ذو مروه فاستوي`
- char_accuracy: 0.9091
- edit_distance: 1
- len_delta: 1

### hf_quran_md_ayah_route_muhsin_al_qasim_100_004_184486
- reciter: `muhsin_al_qasim`
- gold: `فاثرن به نقعا`
- pred: `فاثرنا به نقعا`
- char_accuracy: 0.9091
- edit_distance: 1
- len_delta: 1

### hf_quran_md_ayah_route_muhsin_al_qasim_100_005_184516
- reciter: `muhsin_al_qasim`
- gold: `فوسطن به جمعا`
- pred: `ووسطن به جمعا`
- char_accuracy: 0.9091
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_ghamadi_061_004_154992
- reciter: `ghamadi`
- gold: `ان الله يحب الذين يقاتلون في سبيله صفا كانهم بنيان مرصوص`
- pred: `ان الله يحب الذين يقاتلون في سبيله صفا صفا كانهم بنيان مصوص`
- char_accuracy: 0.9130
- edit_distance: 4
- len_delta: 2

### hf_quran_md_ayah_route_ghamadi_053_051_145032
- reciter: `ghamadi`
- gold: `وثمود فما ابقي`
- pred: `عثمود فما ابقي`
- char_accuracy: 0.9167
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_077_033_169635
- reciter: `abu_bakr_ash_shaatree`
- gold: `كانه جمالت صفر`
- pred: `كانه جماله صفر`
- char_accuracy: 0.9167
- edit_distance: 1
- len_delta: 0

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

### hf_quran_md_ayah_route_muhsin_al_qasim_091_012_181636
- reciter: `muhsin_al_qasim`
- gold: `اذ انبعث اشقاها`
- pred: `اذن بعث اشقاها`
- char_accuracy: 0.9231
- edit_distance: 1
- len_delta: -1

### hf_quran_md_ayah_route_muhsin_al_qasim_092_010_182026
- reciter: `muhsin_al_qasim`
- gold: `فسنيسره للعسري`
- pred: `فسنيسره لالعسري`
- char_accuracy: 0.9231
- edit_distance: 1
- len_delta: 1

### hf_quran_md_ayah_route_muhsin_al_qasim_101_010_184996
- reciter: `muhsin_al_qasim`
- gold: `وما ادراك ما هيه`
- pred: `وما ادراك ما هي`
- char_accuracy: 0.9231
- edit_distance: 1
- len_delta: -1

### hf_quran_md_ayah_route_minshawy_mujawwad_007_080_030992
- reciter: `minshawy_mujawwad`
- gold: `ولوطا اذ قال لقومه اتاتون الفاحشه ما سبقكم بها من احد من العالمين`
- pred: `ولوطا ان قال لقوم ذي اتاتون الفاحشه ما سبقكم بذا من احد من العالمين`
- char_accuracy: 0.9245
- edit_distance: 4
- len_delta: 1

### hf_quran_md_ayah_route_ghamadi_055_039_148182
- reciter: `ghamadi`
- gold: `فيومئذ لا يسال عن ذنبه انس ولا جان`
- pred: `فيومئذ لا يسال عن دان به انس ولا جان`
- char_accuracy: 0.9259
- edit_distance: 2
- len_delta: 1

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_079_043_172635
- reciter: `abu_bakr_ash_shaatree`
- gold: `فيم انت من ذكراها`
- pred: `فيما انت من ذكراها`
- char_accuracy: 0.9286
- edit_distance: 1
- len_delta: 1

### hf_quran_md_ayah_route_warsh_husary_084_001_176534
- reciter: `warsh_husary`
- gold: `اذا السماء انشقت`
- pred: `اذا السماء الشقت`
- char_accuracy: 0.9286
- edit_distance: 1
- len_delta: 0

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

### hf_quran_md_ayah_route_muhsin_al_qasim_109_006_186376
- reciter: `muhsin_al_qasim`
- gold: `لكم دينكم ولي دين`
- pred: `لاكم دينكم ولي دين`
- char_accuracy: 0.9286
- edit_distance: 1
- len_delta: 1

### hf_quran_md_ayah_route_abdul_basit_murattal_021_089_077137
- reciter: `abdul_basit_murattal`
- gold: `وزكريا اذ نادي ربه رب لا تذرني فردا وانت خير الوارثين`
- pred: `وزكريا اب نادي ربه رب لا تذرني فردا وانت خير وارثين`
- char_accuracy: 0.9302
- edit_distance: 3
- len_delta: -2

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_087_017_178935
- reciter: `abu_bakr_ash_shaatree`
- gold: `والاخره خير وابقي`
- pred: `والافره خير وابقي`
- char_accuracy: 0.9333
- edit_distance: 1
- len_delta: 0

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

### hf_quran_md_ayah_route_abdul_basit_murattal_026_080_090337
- reciter: `abdul_basit_murattal`
- gold: `واذا مرضت فهو يشفين`
- pred: `واذا مرت فهو يشفين`
- char_accuracy: 0.9375
- edit_distance: 1
- len_delta: -1

### hf_quran_md_ayah_route_hussary.teacher_035_021_110409
- reciter: `hussary.teacher`
- gold: `ولا الظل ولا الحرور`
- pred: `ولا الظل ولا الحظور`
- char_accuracy: 0.9375
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_083_025_176175
- reciter: `abu_bakr_ash_shaatree`
- gold: `يسقون من رحيق مختوم`
- pred: `يسقون من وحيق مختوم`
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
- pred: `ولا تخزن يوم يبعثون`
- char_accuracy: 0.9412
- edit_distance: 1
- len_delta: -1

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

### hf_quran_md_ayah_route_saood_ash_shuraym_002_278_008520
- reciter: `saood_ash_shuraym`
- gold: `يا ايها الذين امنوا اتقوا الله وذروا ما بقي من الربا ان كنتم مؤمنين`
- pred: `يا ايها الذين امنوا اتقوا الله وادروا ما بقي من الربي ان كنتم مؤمنين`
- char_accuracy: 0.9444
- edit_distance: 3
- len_delta: 1

### hf_quran_md_ayah_route_minshawy_mujawwad_007_139_032762
- reciter: `minshawy_mujawwad`
- gold: `ان هؤلاء متبر ما هم فيه وباطل ما كانوا يعملون`
- pred: `ان هؤلاء متبر ما هم في ذي وباطل ما كانوا يعملون`
- char_accuracy: 0.9444
- edit_distance: 2
- len_delta: 1

### hf_quran_md_ayah_route_abdul_basit_murattal_026_203_094027
- reciter: `abdul_basit_murattal`
- gold: `فيقولوا هل نحن منظرون`
- pred: `فيقولوا هل نحن موظرون`
- char_accuracy: 0.9444
- edit_distance: 1
- len_delta: 0

### hf_quran_md_ayah_route_ghamadi_053_019_144072
- reciter: `ghamadi`
- gold: `افرايتم اللات والعزي`
- pred: `افرايتم الا تو العزي`
- char_accuracy: 0.9444
- edit_distance: 1
- len_delta: -1

### hf_quran_md_ayah_route_warsh_yassin_025_012_085986
- reciter: `warsh_yassin`
- gold: `اذا راتهم من مكان بعيد سمعوا لها تغيظا وزفيرا`
- pred: `اذا راتهم من مكان بعيد سمعوا لها تظيغا وزفيرا`
- char_accuracy: 0.9459
- edit_distance: 2
- len_delta: 0

### hf_quran_md_ayah_route_saood_ash_shuraym_002_018_000720
- reciter: `saood_ash_shuraym`
- gold: `صم بكم عمي فهم لا يرجعون`
- pred: `صم بكم عم فهم لا يرجعون`
- char_accuracy: 0.9474
- edit_distance: 1
- len_delta: -1

### hf_quran_md_ayah_route_warsh_yassin_023_082_082626
- reciter: `warsh_yassin`
- gold: `قالوا ااذا متنا وكنا ترابا وعظاما اانا لمبعوثون`
- pred: `قالوا ايذا متنا وكنا ترابا وعظاما انا لمبعوثون`
- char_accuracy: 0.9500
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
- pred: `لا تري فيها عوجا ولا امت`
- char_accuracy: 0.9500
- edit_distance: 1
- len_delta: -1

### hf_quran_md_ayah_route_hussary.teacher_037_069_115689
- reciter: `hussary.teacher`
- gold: `انهم الفوا اباءهم ضالين`
- pred: `انهم الف واباءهم ضالين`
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

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_079_034_172365
- reciter: `abu_bakr_ash_shaatree`
- gold: `فاذا جاءت الطامه الكبري`
- pred: `فاذا جاءت الطانه الكبري`
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

### hf_quran_md_ayah_route_alafasy_038_040_120280
- reciter: `alafasy`
- gold: `وان له عندنا لزلفي وحسن ماب`
- pred: `وان له عندنا لزلفا وحسن ماب`
- char_accuracy: 0.9545
- edit_distance: 1
- len_delta: 0
