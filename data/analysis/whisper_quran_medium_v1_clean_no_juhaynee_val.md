# Whisper Quran ASR evaluation

## Summary

| metric | value |
|---|---:|
| model_dir | C:\Users\anis\Desktop\tajweed-modular-assessment\checkpoints\content_asr_whisper_medium_quran_v1_clean_no_juhaynee |
| manifest | C:\Users\anis\Desktop\tajweed-modular-assessment\data\manifests\content_v6a_short_hf_ayah_r1_hf_ayah_clean_no_juhaynee.jsonl |
| split | val |
| samples | 407 |
| exact_norm_rate | 0.6953 |
| exact_compact_rate | 0.7076 |
| avg_char_accuracy | 0.9564 |
| cer | 0.0247 |
| wer | 0.0693 |
| avg_gold_char_len | 27.7617 |
| avg_pred_char_len | 27.7371 |

## Worst examples

### hf_quran_md_ayah_route_saood_ash_shuraym_002_001_000210
- gold: `الم`
- pred: `الافلاميم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 6/3

### hf_quran_md_ayah_route_saood_ash_shuraym_003_001_008790
- gold: `الم`
- pred: `الافلاميم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 6/3

### hf_quran_md_ayah_route_ali_jaber_028_001_097568
- gold: `طسم`
- pred: `باسيم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 3/3

### hf_quran_md_ayah_route_ali_jaber_029_001_100208
- gold: `الم`
- pred: `الفلاميم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 5/3

### hf_quran_md_ayah_route_ali_jaber_030_001_102278
- gold: `الم`
- pred: `الفلاميم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 5/3

### hf_quran_md_ayah_route_ali_jaber_031_001_104078
- gold: `الم`
- pred: `الفلاميم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 5/3

### hf_quran_md_ayah_route_ali_jaber_032_001_105098
- gold: `الم`
- pred: `الفلاميم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 5/3

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

### hf_quran_md_ayah_route_abdul_basit_murattal_026_001_087967
- gold: `طسم`
- pred: `طاسيم`
- char_accuracy: 0.333
- CER contribution edit/gold_len: 2/3

### hf_quran_md_ayah_route_husary_mujawwad_012_099_050824
- gold: `فلما دخلوا علي يوسف اوي اليه ابويه وقال ادخلوا مصر ان شاء الله امنين`
- pred: `فلما دخلوا علي يوسف اواء له ابوه`
- char_accuracy: 0.436
- CER contribution edit/gold_len: 31/55

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

### hf_quran_md_ayah_route_muhsin_al_qasim_109_006_186376
- gold: `لكم دينكم ولي دين`
- pred: `اوانكم دينكم ولي دين`
- char_accuracy: 0.714
- CER contribution edit/gold_len: 4/14

### hf_quran_md_ayah_route_minshawy_mujawwad_007_139_032762
- gold: `ان هؤلاء متبر ما هم فيه وباطل ما كانوا يعملون`
- pred: `ان انا انا متبغ ما هم فيذ وباطل ما كانوا يعملون`
- char_accuracy: 0.806
- CER contribution edit/gold_len: 7/36

### hf_quran_md_ayah_route_ali_jaber_031_003_104138
- gold: `هدي ورحمة للمحسنين`
- pred: `عودا ورحمة للمحسنين`
- char_accuracy: 0.812
- CER contribution edit/gold_len: 3/16

### hf_quran_md_ayah_route_muhsin_al_qasim_096_007_183376
- gold: `ان راه استغني`
- pred: `قراه استغني`
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

### hf_quran_md_ayah_route_abdul_basit_murattal_026_080_090337
- gold: `واذا مرضت فهو يشفين`
- pred: `واذا مرت فهو يشكين`
- char_accuracy: 0.875
- CER contribution edit/gold_len: 2/16

### hf_quran_md_ayah_route_husary_mujawwad_012_079_050224
- gold: `قال معاذ الله ان ناخذ الا من وجدنا متاعنا عنده انا اذا لظالمون`
- pred: `قال معاذ الله ان ناخذ الا من وجدنا متاعنا عنده انا اذا لغبنا`
- char_accuracy: 0.880
- CER contribution edit/gold_len: 6/50

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_080_034_173745
- gold: `يوم يفر المرء من اخيه`
- pred: `يوم يفروا المرء من اخيه`
- char_accuracy: 0.882
- CER contribution edit/gold_len: 2/17

### hf_quran_md_ayah_route_muhsin_al_qasim_107_002_185956
- gold: `فذلك الذي يدع اليتيم`
- pred: `وذلك الذي يدعي اليتيم`
- char_accuracy: 0.882
- CER contribution edit/gold_len: 2/17

### hf_quran_md_ayah_route_ghamadi_053_019_144072
- gold: `افرايتم اللات والعزي`
- pred: `افرايتم الا تول عزي`
- char_accuracy: 0.889
- CER contribution edit/gold_len: 2/18

### hf_quran_md_ayah_route_ghamadi_061_004_154992
- gold: `ان الله يحب الذين يقاتلون في سبيله صفا كانهم بنيان مرصوص`
- pred: `ان الله يحب الذين يقاتلون في سبيله صفا صفا كانهم بنيان مقصوس`
- char_accuracy: 0.891
- CER contribution edit/gold_len: 5/46

### hf_quran_md_ayah_route_saood_ash_shuraym_002_018_000720
- gold: `صم بكم عمي فهم لا يرجعون`
- pred: `صم بك منعم فهم لا يرجعون`
- char_accuracy: 0.895
- CER contribution edit/gold_len: 2/19

### hf_quran_md_ayah_route_saood_ash_shuraym_002_042_001440
- gold: `ولا تلبسوا الحق بالباطل وتكتموا الحق وانتم تعلمون`
- pred: `ولا تلبس الحق بالباطل وتكتم الحق وانتم تعلمون`
- char_accuracy: 0.905
- CER contribution edit/gold_len: 4/42

### hf_quran_md_ayah_route_ibrahim_akhdar_053_006_143681
- gold: `ذو مرة فاستوي`
- pred: `ذو مروة فاستوي`
- char_accuracy: 0.909
- CER contribution edit/gold_len: 1/11

### hf_quran_md_ayah_route_muhsin_al_qasim_092_004_181846
- gold: `ان سعيكم لشتي`
- pred: `ان سعيكم لشتا`
- char_accuracy: 0.909
- CER contribution edit/gold_len: 1/11


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

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_167_019771
- gold: `ان الذين كفروا وصدوا عن سبيل الله قد ضلوا ضلالا بعيدا`
- pred: `ان الذين كفروا وصدوا عن سبيل الله قد ضلوا ضلالا بعيدا`
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

### hf_quran_md_ayah_route_minshawy_mujawwad_007_109_031862
- gold: `قال الملا من قوم فرعون ان هذا لساحر عليم`
- pred: `قال الملا من قوم فرعون ان هذا لساحر عليم`
- char_accuracy: 1.000
