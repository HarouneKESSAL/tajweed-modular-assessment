# Whisper Quran ASR evaluation

## Summary

| metric | value |
|---|---:|
| model_dir | C:\Users\anis\Desktop\tajweed-modular-assessment\checkpoints\content_asr_whisper_small_quran_v1_clean_no_juhaynee |
| manifest | C:\Users\anis\Desktop\tajweed-modular-assessment\data\manifests\content_v6a_short_hf_ayah_r1_hf_ayah_clean_no_juhaynee.jsonl |
| split | val |
| samples | 407 |
| exact_norm_rate | 0.5799 |
| exact_compact_rate | 0.6069 |
| avg_char_accuracy | 0.9486 |
| cer | 0.0331 |
| wer | 0.1173 |
| avg_gold_char_len | 27.7617 |
| avg_pred_char_len | 27.7248 |

## Worst examples

### hf_quran_md_ayah_route_saood_ash_shuraym_002_001_000210
- gold: `الم`
- pred: `فالاخلام ميم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 8/3

### hf_quran_md_ayah_route_saood_ash_shuraym_003_001_008790
- gold: `الم`
- pred: `الفلام ميم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 6/3

### hf_quran_md_ayah_route_ali_jaber_029_001_100208
- gold: `الم`
- pred: `الفلامم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 4/3

### hf_quran_md_ayah_route_ali_jaber_030_001_102278
- gold: `الم`
- pred: `الفلام ميم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 6/3

### hf_quran_md_ayah_route_ali_jaber_031_001_104078
- gold: `الم`
- pred: `الفلامم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 4/3

### hf_quran_md_ayah_route_ali_jaber_032_001_105098
- gold: `الم`
- pred: `الفلاميم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 5/3

### hf_quran_md_ayah_route_hussary.teacher_036_001_111159
- gold: `يس`
- pred: `ياسين`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 3/2

### hf_quran_md_ayah_route_alafasy_042_002_128200
- gold: `عسق`
- pred: `عنس قاف`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 3/3

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

### hf_quran_md_ayah_route_ali_jaber_028_001_097568
- gold: `طسم`
- pred: `طاسيم`
- char_accuracy: 0.333
- CER contribution edit/gold_len: 2/3

### hf_quran_md_ayah_route_husary_mujawwad_012_099_050824
- gold: `فلما دخلوا علي يوسف اوي اليه ابويه وقال ادخلوا مصر ان شاء الله امنين`
- pred: `فلما دخلوا علي يوسفا واليه ابوي`
- char_accuracy: 0.473
- CER contribution edit/gold_len: 29/55

### hf_quran_md_ayah_route_husary_mujawwad_012_089_050524
- gold: `قال هل علمتم ما فعلتم بيوسف واخيه اذ انتم جاهلون`
- pred: `قال هل علمتم ما فعلتم بيوسف واخيين`
- char_accuracy: 0.692
- CER contribution edit/gold_len: 12/39

### hf_quran_md_ayah_route_minshawy_mujawwad_006_041_024872
- gold: `بل اياه تدعون فيكشف ما تدعون اليه ان شاء وتنسون ما تشركون`
- pred: `بل اياه تدعون فيكشف ما تدعون اليه انشاء`
- char_accuracy: 0.696
- CER contribution edit/gold_len: 14/46

### hf_quran_md_ayah_route_abdul_basit_murattal_027_021_095377
- gold: `لاعذبنه عذابا شديدا او لاذبحنه او لياتيني بسلطان مبين`
- pred: `ذا اعذبنه عذابا شديدا اول اذبحنه اول اذبحنه اول ياتيني بسرطان مبين`
- char_accuracy: 0.733
- CER contribution edit/gold_len: 12/45

### hf_quran_md_ayah_route_ghamadi_053_019_144072
- gold: `افرايتم اللات والعزي`
- pred: `افرايتم لا تول عزا`
- char_accuracy: 0.778
- CER contribution edit/gold_len: 4/18

### hf_quran_md_ayah_route_muhsin_al_qasim_102_004_185146
- gold: `ثم كلا سوف تعلمون`
- pred: `واما كلا سوف تعلمون`
- char_accuracy: 0.786
- CER contribution edit/gold_len: 3/14

### hf_quran_md_ayah_route_warsh_yassin_023_007_080376
- gold: `فمن ابتغي وراء ذلك فاولئك هم العادون`
- pred: `فمن اذ اغاري اذلك فاولئك هم العادون`
- char_accuracy: 0.800
- CER contribution edit/gold_len: 6/30

### hf_quran_md_ayah_route_ali_jaber_031_003_104138
- gold: `هدي ورحمة للمحسنين`
- pred: `عودا ورحمة للمحسنين`
- char_accuracy: 0.812
- CER contribution edit/gold_len: 3/16

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_087_009_178695
- gold: `فذكر ان نفعت الذكري`
- pred: `فذكر ان مفعت ذكري`
- char_accuracy: 0.812
- CER contribution edit/gold_len: 3/16

### hf_quran_md_ayah_route_ibrahim_akhdar_053_006_143681
- gold: `ذو مرة فاستوي`
- pred: `ذو مروت فاستوي`
- char_accuracy: 0.818
- CER contribution edit/gold_len: 2/11

### hf_quran_md_ayah_route_husary_mujawwad_012_079_050224
- gold: `قال معاذ الله ان ناخذ الا من وجدنا متاعنا عنده انا اذا لظالمون`
- pred: `قال معاذ الله ان اخذ الا من وجدنا متاعنا عنده انا اذا لغل من المنزلين`
- char_accuracy: 0.820
- CER contribution edit/gold_len: 9/50

### hf_quran_md_ayah_route_ghamadi_053_051_145032
- gold: `وثمود فما ابقي`
- pred: `عث مود فما ابقا`
- char_accuracy: 0.833
- CER contribution edit/gold_len: 2/12

### hf_quran_md_ayah_route_saood_ash_shuraym_002_018_000720
- gold: `صم بكم عمي فهم لا يرجعون`
- pred: `طمم بك من عمي فهم لا يرجعون`
- char_accuracy: 0.842
- CER contribution edit/gold_len: 3/19

### hf_quran_md_ayah_route_muhsin_al_qasim_091_012_181636
- gold: `اذ انبعث اشقاها`
- pred: `اذا بعت اشقاها`
- char_accuracy: 0.846
- CER contribution edit/gold_len: 2/13

### hf_quran_md_ayah_route_ibrahim_akhdar_053_003_143591
- gold: `وما ينطق عن الهوي`
- pred: `وما ينطق علي الهوي`
- char_accuracy: 0.857
- CER contribution edit/gold_len: 2/14

### hf_quran_md_ayah_route_muhsin_al_qasim_091_004_181396
- gold: `والليل اذا يغشاها`
- pred: `الليل اذا يغشيها`
- char_accuracy: 0.867
- CER contribution edit/gold_len: 2/15

### hf_quran_md_ayah_route_muhsin_al_qasim_104_002_185416
- gold: `الذي جمع مالا وعدده`
- pred: `والذي جمع مالا وعددة`
- char_accuracy: 0.875
- CER contribution edit/gold_len: 2/16

### hf_quran_md_ayah_route_ghamadi_056_025_150102
- gold: `لا يسمعون فيها لغوا ولا تاثيما`
- pred: `فيسمعون فيها لغوا ولا تافيما`
- char_accuracy: 0.880
- CER contribution edit/gold_len: 3/25

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_078_006_170325
- gold: `الم نجعل الارض مهادا`
- pred: `الم نجعل الارض مهدي`
- char_accuracy: 0.882
- CER contribution edit/gold_len: 2/17


## Best examples

### hf_quran_md_ayah_route_saood_ash_shuraym_002_004_000300
- gold: `والذين يؤمنون بما انزل اليك وما انزل من قبلك وبالاخرة هم يوقنون`
- pred: `والذين يؤمنون بما انزل اليك وما انزل من قبلك وبالاخرة هم يوقنون`
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
- pred: `يا ايها الذين امنوا اتقوا الله حقت قاته ولا تموتن الا وانتم مسلمون`
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

### hf_quran_md_ayah_route_minshawy_mujawwad_007_025_029342
- gold: `قال فيها تحيون وفيها تموتون ومنها تخرجون`
- pred: `قال فيها تحيون وفيها تموتون ومنها تخرجون`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_minshawy_mujawwad_007_109_031862
- gold: `قال الملا من قوم فرعون ان هذا لساحر عليم`
- pred: `قال الملا من قوم فرعون ان هذا لساحر عليم`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_minshawy_mujawwad_007_174_033812
- gold: `وكذلك نفصل الايات ولعلهم يرجعون`
- pred: `وكذلك نفصل الايات ولعلهم يرجعون`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdullah_basfar_008_055_036423
- gold: `ان شر الدواب عند الله الذين كفروا فهم لا يؤمنون`
- pred: `ان شر الدواب عند الله الذين كفروا فهم لا يؤمنون`
- char_accuracy: 1.000
