# Whisper Quran ASR evaluation

## Summary

| metric | value |
|---|---:|
| model_dir | C:\Users\anis\Desktop\tajweed-modular-assessment\checkpoints\content_asr_whisper_medium_quran_v2_weighted |
| manifest | C:\Users\anis\Desktop\tajweed-modular-assessment\data\manifests\content_v6a_short_hf_ayah_r1_hf_ayah_clean_no_juhaynee.jsonl |
| split | val |
| samples | 407 |
| exact_norm_rate | 0.6978 |
| exact_compact_rate | 0.7174 |
| avg_char_accuracy | 0.9644 |
| cer | 0.0215 |
| wer | 0.0716 |
| avg_gold_char_len | 27.7617 |
| avg_pred_char_len | 27.7174 |

## Worst examples

### hf_quran_md_ayah_route_saood_ash_shuraym_002_001_000210
- gold: `الم`
- pred: `الافلاميم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 6/3

### hf_quran_md_ayah_route_abdul_basit_murattal_026_001_087967
- gold: `طسم`
- pred: `طاسيميم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 4/3

### hf_quran_md_ayah_route_ali_jaber_028_001_097568
- gold: `طسم`
- pred: `باسيم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 3/3

### hf_quran_md_ayah_route_hussary.teacher_036_001_111159
- gold: `يس`
- pred: `يا سين`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 3/2

### hf_quran_md_ayah_route_alafasy_042_002_128200
- gold: `عسق`
- pred: `عين سينقاف`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 6/3

### hf_quran_md_ayah_route_hussary.teacher_037_152_118179
- gold: `ولد الله وانهم لكاذبون`
- pred: `من افكهم ليقولون ولد الله وانهم لكاذبون`
- char_accuracy: 0.263
- CER contribution edit/gold_len: 14/19

### hf_quran_md_ayah_route_saood_ash_shuraym_003_001_008790
- gold: `الم`
- pred: `الميم`
- char_accuracy: 0.333
- CER contribution edit/gold_len: 2/3

### hf_quran_md_ayah_route_ali_jaber_030_001_102278
- gold: `الم`
- pred: `الميم`
- char_accuracy: 0.333
- CER contribution edit/gold_len: 2/3

### hf_quran_md_ayah_route_ali_jaber_032_001_105098
- gold: `الم`
- pred: `الميم`
- char_accuracy: 0.333
- CER contribution edit/gold_len: 2/3

### hf_quran_md_ayah_route_husary_mujawwad_012_099_050824
- gold: `فلما دخلوا علي يوسف اوي اليه ابويه وقال ادخلوا مصر ان شاء الله امنين`
- pred: `فلما دخلوا علي يوسف اوي اليه ابو يوسف`
- char_accuracy: 0.509
- CER contribution edit/gold_len: 27/55

### hf_quran_md_ayah_route_husary_mujawwad_012_089_050524
- gold: `قال هل علمتم ما فعلتم بيوسف واخيه اذ انتم جاهلون`
- pred: `قال هل علمتم ما فعلتم بيوسف واخي`
- char_accuracy: 0.667
- CER contribution edit/gold_len: 13/39

### hf_quran_md_ayah_route_minshawy_mujawwad_006_041_024872
- gold: `بل اياه تدعون فيكشف ما تدعون اليه ان شاء وتنسون ما تشركون`
- pred: `بل اياه تدعون فيكشف ما تدعون اليه ان شاء`
- char_accuracy: 0.696
- CER contribution edit/gold_len: 14/46

### hf_quran_md_ayah_route_muhsin_al_qasim_096_007_183376
- gold: `ان راه استغني`
- pred: `ارواه استغني`
- char_accuracy: 0.818
- CER contribution edit/gold_len: 2/11

### hf_quran_md_ayah_route_muhsin_al_qasim_096_018_183706
- gold: `سندع الزبانية`
- pred: `وسندعو الزبانية`
- char_accuracy: 0.833
- CER contribution edit/gold_len: 2/12

### hf_quran_md_ayah_route_muhsin_al_qasim_091_004_181396
- gold: `والليل اذا يغشاها`
- pred: `الليل اذا يغشانها`
- char_accuracy: 0.867
- CER contribution edit/gold_len: 2/15

### hf_quran_md_ayah_route_warsh_yassin_023_009_080436
- gold: `والذين هم علي صلواتهم يحافظون`
- pred: `والذين هم علي صلواته ابن يحافظون`
- char_accuracy: 0.880
- CER contribution edit/gold_len: 3/25

### hf_quran_md_ayah_route_abdul_basit_murattal_026_199_093907
- gold: `فقراه عليهم ما كانوا به مؤمنين`
- pred: `فقر ابو عليه ما كانوا به مؤمنين`
- char_accuracy: 0.880
- CER contribution edit/gold_len: 3/25

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_080_034_173745
- gold: `يوم يفر المرء من اخيه`
- pred: `يوم يفروا المرء من اخيه`
- char_accuracy: 0.882
- CER contribution edit/gold_len: 2/17

### hf_quran_md_ayah_route_muhsin_al_qasim_107_002_185956
- gold: `فذلك الذي يدع اليتيم`
- pred: `وذلك الذي يدعو اليتيم`
- char_accuracy: 0.882
- CER contribution edit/gold_len: 2/17

### hf_quran_md_ayah_route_husary_mujawwad_012_079_050224
- gold: `قال معاذ الله ان ناخذ الا من وجدنا متاعنا عنده انا اذا لظالمون`
- pred: `قال معاذ الله ان ناخذ الا من وجدنا متاعنا عنده انا اذا لغبر معا`
- char_accuracy: 0.900
- CER contribution edit/gold_len: 5/50

### hf_quran_md_ayah_route_saood_ash_shuraym_002_042_001440
- gold: `ولا تلبسوا الحق بالباطل وتكتموا الحق وانتم تعلمون`
- pred: `ولا تلبس الحق بالباطل وتكتم الحق وانتم تعلمون`
- char_accuracy: 0.905
- CER contribution edit/gold_len: 4/42

### hf_quran_md_ayah_route_warsh_husary_076_025_168464
- gold: `واذكر اسم ربك بكرة واصيلا`
- pred: `والكرس مربك بكرة واصيلا`
- char_accuracy: 0.905
- CER contribution edit/gold_len: 2/21

### hf_quran_md_ayah_route_ibrahim_akhdar_053_006_143681
- gold: `ذو مرة فاستوي`
- pred: `ذو مروة فاستوي`
- char_accuracy: 0.909
- CER contribution edit/gold_len: 1/11

### hf_quran_md_ayah_route_muhsin_al_qasim_100_004_184486
- gold: `فاثرن به نقعا`
- pred: `فاثرنا به نقعا`
- char_accuracy: 0.909
- CER contribution edit/gold_len: 1/11

### hf_quran_md_ayah_route_muhsin_al_qasim_100_005_184516
- gold: `فوسطن به جمعا`
- pred: `ووسطن به جمعا`
- char_accuracy: 0.909
- CER contribution edit/gold_len: 1/11

### hf_quran_md_ayah_route_ghamadi_061_004_154992
- gold: `ان الله يحب الذين يقاتلون في سبيله صفا كانهم بنيان مرصوص`
- pred: `ان الله يحب الذين يقاتلون في سبيله صفا صفا كانهم بنيان مصوص`
- char_accuracy: 0.913
- CER contribution edit/gold_len: 4/46

### hf_quran_md_ayah_route_ghamadi_053_051_145032
- gold: `وثمود فما ابقي`
- pred: `عثمود فما ابقي`
- char_accuracy: 0.917
- CER contribution edit/gold_len: 1/12

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_077_033_169635
- gold: `كانه جمالت صفر`
- pred: `كانه جمالة صفر`
- char_accuracy: 0.917
- CER contribution edit/gold_len: 1/12

### hf_quran_md_ayah_route_muhsin_al_qasim_097_004_183856
- gold: `تنزل الملائكة والروح فيها باذن ربهم من كل امر`
- pred: `نزلوا الملائكة والروح فيها باذن ربهم من كل امر`
- char_accuracy: 0.919
- CER contribution edit/gold_len: 3/37

### hf_quran_md_ayah_route_warsh_husary_077_001_168674
- gold: `والمرسلات عرفا`
- pred: `والموسلات عرفا`
- char_accuracy: 0.923
- CER contribution edit/gold_len: 1/13


## Best examples

### hf_quran_md_ayah_route_saood_ash_shuraym_002_004_000300
- gold: `والذين يؤمنون بما انزل اليك وما انزل من قبلك وبالاخرة هم يوقنون`
- pred: `والذين يؤمنون بما انزل اليك وما انزل من قبلك وبالاخرة هم يوقنون`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_saood_ash_shuraym_002_056_001860
- gold: `ثم بعثناكم من بعد موتكم لعلكم تشكرون`
- pred: `ثم بعثناكم من بعد موتكم لعلكم تشكرون`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_saood_ash_shuraym_002_066_002160
- gold: `فجعلناها نكالا لما بين يديها وما خلفها وموعظة للمتقين`
- pred: `فجعلناها نكالا لما بين يديها وما خلفها وموعظة للمتقين`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_saood_ash_shuraym_002_077_002490
- gold: `اولا يعلمون ان الله يعلم ما يسرون وما يعلنون`
- pred: `اولا يعلمون ان الله يعلم ما يسرون وما يعلنون`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_saood_ash_shuraym_002_179_005550
- gold: `ولكم في القصاص حياة يا اولي الالباب لعلكم تتقون`
- pred: `ولكم في القصاص حياة يا اولي الالباب لعلكم تتقون`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_saood_ash_shuraym_003_082_011220
- gold: `فمن تولي بعد ذلك فاولئك هم الفاسقون`
- pred: `فمن تولي بعد ذلك فاولئك هم الفاسقون`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_saood_ash_shuraym_003_089_011430
- gold: `الا الذين تابوا من بعد ذلك واصلحوا فان الله غفور رحيم`
- pred: `الا الذين تابوا من بعد ذلك واصلحوا فان الله غفور رحيم`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_saood_ash_shuraym_003_102_011820
- gold: `يا ايها الذين امنوا اتقوا الله حق تقاته ولا تموتن الا وانتم مسلمون`
- pred: `يا ايها الذين امنوا اتقوا الله حق تقاته ولا تموتن الا وانتم مسلمون`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_003_131_012691
- gold: `واتقوا النار التي اعدت للكافرين`
- pred: `واتقوا النار التي اعدت للكافرين`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_003_158_013501
- gold: `ولئن متم او قتلتم لالي الله تحشرون`
- pred: `ولئن متم او قتلتم لالي الله تحشرون`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_068_016801
- gold: `ولهديناهم صراطا مستقيما`
- pred: `ولهديناهم صراطا مستقيما`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_071_016891
- gold: `يا ايها الذين امنوا خذوا حذركم فانفروا ثبات او انفروا جميعا`
- pred: `يا ايها الذين امنوا خذوا حذركم فانفروا ثبات او انفروا جميعا`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_112_018121
- gold: `ومن يكسب خطيئة او اثما ثم يرم به بريئا فقد احتمل بهتانا واثما مبينا`
- pred: `ومن يكسب خطيئة او اثما ثم يرم به بريئا فقد احتمل بهتانا واثما مبينا`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_138_018901
- gold: `بشر المنافقين بان لهم عذابا اليما`
- pred: `بشر المنافقين بان لهم عذابا اليما`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_168_019801
- gold: `ان الذين كفروا وظلموا لم يكن الله ليغفر لهم ولا ليهديهم طريقا`
- pred: `ان الذين كفروا وظلموا لم يكن الله ليغفر لهم ولا ليهديهم طريقا`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_005_030_020941
- gold: `فطوعت له نفسه قتل اخيه فقتله فاصبح من الخاسرين`
- pred: `فطوعت له نفسه قتل اخيه فقتله فاصبح من الخاسرين`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_006_009_023911
- gold: `ولو جعلناه ملكا لجعلناه رجلا وللبسنا عليهم ما يلبسون`
- pred: `ولو جعلناه ملكا لجعلناه رجلا وللبسنا عليهم ما يلبسون`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_minshawy_mujawwad_007_005_028742
- gold: `فما كان دعواهم اذ جاءهم باسنا الا ان قالوا انا كنا ظالمين`
- pred: `فما كان دعواهم اذ جاءهم باسنا الا ان قالوا انا كنا ظالمين`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_minshawy_mujawwad_007_174_033812
- gold: `وكذلك نفصل الايات ولعلهم يرجعون`
- pred: `وكذلك نفصل الايات ولعلهم يرجعون`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdullah_basfar_008_055_036423
- gold: `ان شر الدواب عند الله الذين كفروا فهم لا يؤمنون`
- pred: `ان شر الدواب عند الله الذين كفروا فهم لا يؤمنون`
- char_accuracy: 1.000
