# Whisper Quran ASR evaluation

## Summary

| metric | value |
|---|---:|
| model_dir | C:\Users\anis\Desktop\tajweed-modular-assessment\checkpoints\content_asr_whisper_small_quran_v1_full |
| manifest | C:\Users\anis\Desktop\tajweed-modular-assessment\data\manifests\content_v6a_short_hf_ayah_r1_hf_ayah_clean_all.jsonl |
| split | val |
| samples | 448 |
| exact_norm_rate | 0.4754 |
| exact_compact_rate | 0.4933 |
| avg_char_accuracy | 0.8873 |
| cer | 0.0934 |
| wer | 0.2023 |
| avg_gold_char_len | 27.4955 |
| avg_pred_char_len | 27.4821 |

## Worst examples

### hf_quran_md_ayah_route_saood_ash_shuraym_002_001_000210
- gold: `الم`
- pred: `الاخلام ميم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 7/3

### hf_quran_md_ayah_route_ali_jaber_028_001_097568
- gold: `طسم`
- pred: `قاسيم ميم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 6/3

### hf_quran_md_ayah_route_hussary.teacher_036_001_111159
- gold: `يس`
- pred: `ياسين`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 3/2

### hf_quran_md_ayah_route_alafasy_042_002_128200
- gold: `عسق`
- pred: `عن سينقاف`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 5/3

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_069_049_161143
- gold: `وانا لنعلم ان منكم مكذبين`
- pred: `وانا لحقهم ان يكونوا كذبين امامهم ان تدعون جزء جزئين`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 29/21

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_015_165283
- gold: `ثم يطمع ان ازيد`
- pred: `والا اقوم بكتسين`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 13/12

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_050_166333
- gold: `كانهم حمر مستنفرة`
- pred: `فلما نرغب منه تسليم اسرق`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 15/15

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_040_167713
- gold: `اليس ذلك بقادر علي ان يحيي الموتي`
- pred: `علي السباح المتحدة والمجنون منهم ومن يومئذ بلدا`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 31/27

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_008_165073
- gold: `فاذا نقر في الناقور`
- pred: `ايكذاه تقول لهم لا كذبون`
- char_accuracy: 0.062
- CER contribution edit/gold_len: 15/16

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_030_167413
- gold: `الي ربك يومئذ المساق`
- pred: `وما يقولون ليكم في الساعة`
- char_accuracy: 0.118
- CER contribution edit/gold_len: 15/17

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_035_167563
- gold: `ثم اولي لك فاولي`
- pred: `فانه لحول فكبر`
- char_accuracy: 0.154
- CER contribution edit/gold_len: 11/13

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_004_157333
- gold: `ثم ارجع البصر كرتين ينقلب اليك البصر خاسئا وهو حسير`
- pred: `ثم انه لحسرت الله علي من قبل ايها المصرخين فاصبروا صلواتهم واسيرون`
- char_accuracy: 0.167
- CER contribution edit/gold_len: 35/42

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_030_165733
- gold: `عليها تسعة عشر`
- pred: `علينا المستكافرون`
- char_accuracy: 0.167
- CER contribution edit/gold_len: 10/12

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_038_165973
- gold: `كل نفس بما كسبت رهينة`
- pred: `ان يوسي بحثنا تسريها`
- char_accuracy: 0.176
- CER contribution edit/gold_len: 14/17

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_012_163753
- gold: `وانا ظننا ان لن نعجز الله في الارض ولن نعجزه هربا`
- pred: `وما ادراه ما قالت ان يسروا علي ربهم ويمكني سروا رابا`
- char_accuracy: 0.179
- CER contribution edit/gold_len: 32/39

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_068_047_159523
- gold: `ام عندهم الغيب فهم يكتبون`
- pred: `والذين ادبروا بيوم القضون`
- char_accuracy: 0.190
- CER contribution edit/gold_len: 17/21

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_073_014_164653
- gold: `يوم ترجف الارض والجبال وكانت الجبال كثيبا مهيلا`
- pred: `فتذر من ربهم جمال ويجعلهم بما تريدون محشاب`
- char_accuracy: 0.200
- CER contribution edit/gold_len: 32/40

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_009_166783
- gold: `وجمع الشمس والقمر`
- pred: `والذين هم من المزيد`
- char_accuracy: 0.200
- CER contribution edit/gold_len: 12/15

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_022_167173
- gold: `وجوه يومئذ ناضرة`
- pred: `عن نميه يومئذ بلا طعام`
- char_accuracy: 0.214
- CER contribution edit/gold_len: 11/14

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_069_005_159823
- gold: `فاما ثمود فاهلكوا بالطاغية`
- pred: `قال انا لسه ربهم نحن ملطمين`
- char_accuracy: 0.217
- CER contribution edit/gold_len: 18/23

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_025_163303
- gold: `مما خطيئاتهم اغرقوا فادخلوا نارا فلم يجدوا لهم من دون الله انصارا`
- pred: `انها فقل ان ياتيهم بمجردهم اولئا واحدة ومن المزيد منهم مصبحين`
- char_accuracy: 0.222
- CER contribution edit/gold_len: 42/54

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_043_166123
- gold: `قالوا لم نك من المصلين`
- pred: `والذين هم من شرقهم`
- char_accuracy: 0.222
- CER contribution edit/gold_len: 14/18

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_037_167623
- gold: `الم يك نطفة من مني يمني`
- pred: `عليه وطبقه فجلسنا`
- char_accuracy: 0.222
- CER contribution edit/gold_len: 14/18

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_011_157543
- gold: `فاعترفوا بذنبهم فسحقا لاصحاب السعير`
- pred: `ايعتبروا منهم ان نساعد بمعفوه مساءلين`
- char_accuracy: 0.226
- CER contribution edit/gold_len: 24/31

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_064_005_156103
- gold: `الم ياتكم نبا الذين كفروا من قبل فذاقوا وبال امرهم ولهم عذاب اليم`
- pred: `وانه كان لكم مخيمين ومن يومئذ كنا نحن بمعاري احد منهم جعلوا تلي`
- char_accuracy: 0.226
- CER contribution edit/gold_len: 41/53

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_031_162163
- gold: `فمن ابتغي وراء ذلك فاولئك هم العادون`
- pred: `ثم ادراك وطعاما علي تجهياتهم ما هرون`
- char_accuracy: 0.233
- CER contribution edit/gold_len: 23/30

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_014_157633
- gold: `الا يعلم من خلق وهو اللطيف الخبير`
- pred: `بلا لحق القضوة ولم يدعوه فليسل`
- char_accuracy: 0.259
- CER contribution edit/gold_len: 20/27

### hf_quran_md_ayah_route_hussary.teacher_037_152_118179
- gold: `ولد الله وانهم لكاذبون`
- pred: `من افكهم ليقولون ولد الله وانهم لكاذبون`
- char_accuracy: 0.263
- CER contribution edit/gold_len: 14/19

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_068_034_159133
- gold: `ان للمتقين عند ربهم جنات النعيم`
- pred: `ان لن تكذب ومن يترك بيوم احدا مليم`
- char_accuracy: 0.269
- CER contribution edit/gold_len: 19/26

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_020_165433
- gold: `ثم قتل كيف قدر`
- pred: `وانه تدري في وقدر`
- char_accuracy: 0.273
- CER contribution edit/gold_len: 8/11


## Best examples

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

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_068_016801
- gold: `ولهديناهم صراطا مستقيما`
- pred: `ولهديناهم صراطا مستقيما`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_138_018901
- gold: `بشر المنافقين بان لهم عذابا اليما`
- pred: `بشر المنافقين بان لهم عذابا اليما`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_005_030_020941
- gold: `فطوعت له نفسه قتل اخيه فقتله فاصبح من الخاسرين`
- pred: `فطوعت له نفسه قتل اخيه فقتله فاصبح من الخاسرين`
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

### hf_quran_md_ayah_route_abdullah_basfar_010_086_043473
- gold: `ونجنا برحمتك من القوم الكافرين`
- pred: `ونجنا برحمتك من القوم الكافرين`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdullah_basfar_011_079_046533
- gold: `قالوا لقد علمت ما لنا في بناتك من حق وانك لتعلم ما نريد`
- pred: `قالوا لقد علمت ما لنا في بناتك من حق وانك لتعلم ما نريد`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdullah_basfar_011_115_047613
- gold: `واصبر فان الله لا يضيع اجر المحسنين`
- pred: `واصبر فان الله لا يضيع اجر المحسنين`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_husary_mujawwad_012_058_049594
- gold: `وجاء اخوة يوسف فدخلوا عليه فعرفهم وهم له منكرون`
- pred: `وجاء اخوة يوسف فدخلوا عليه فعرفهم وهم له منكرون`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_husary_mujawwad_012_074_050074
- gold: `قالوا فما جزاؤه ان كنتم كاذبين`
- pred: `قالوا فما جزاؤه ان كنتم كاذبين`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_husary_mujawwad_012_095_050704
- gold: `قالوا تالله انك لفي ضلالك القديم`
- pred: `قالوا تالله انك لفي ضلالك القديم`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_husary_mujawwad_015_045_055384
- gold: `ان المتقين في جنات وعيون`
- pred: `ان المتقين في جنات وعيون`
- char_accuracy: 1.000
