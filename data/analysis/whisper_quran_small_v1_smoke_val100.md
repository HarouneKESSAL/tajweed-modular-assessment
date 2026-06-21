# Whisper Quran ASR evaluation

## Summary

| metric | value |
|---|---:|
| model_dir | C:\Users\anis\Desktop\tajweed-modular-assessment\checkpoints\content_asr_whisper_small_quran_v1_smoke |
| manifest | C:\Users\anis\Desktop\tajweed-modular-assessment\data\manifests\content_v6a_short_hf_ayah_r1_hf_ayah_clean_all.jsonl |
| split | val |
| samples | 100 |
| exact_norm_rate | 0.1200 |
| exact_compact_rate | 0.1600 |
| avg_char_accuracy | 0.8987 |
| cer | 0.1086 |
| wer | 0.2941 |
| avg_gold_char_len | 36.8300 |
| avg_pred_char_len | 36.7600 |

## Worst examples

### hf_quran_md_ayah_route_saood_ash_shuraym_002_001_000210
- gold: `الم`
- pred: `الفلامم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 4/3

### hf_quran_md_ayah_route_saood_ash_shuraym_003_001_008790
- gold: `الم`
- pred: `الفلامم`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 4/3

### hf_quran_md_ayah_route_abdullah_basfar_009_057_038733
- gold: `لو يجدون ملجا او مغارات او مدخلا لولوا اليه وهم يجمحون`
- pred: `لو يجدون ملجاااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااااا`
- char_accuracy: 0.000
- CER contribution edit/gold_len: 112/44

### hf_quran_md_ayah_route_husary_mujawwad_012_099_050824
- gold: `فلما دخلوا علي يوسف اوي اليه ابويه وقال ادخلوا مصر ان شاء الله امنين`
- pred: `فلما دخلوا علي جوس فاوائليه ابوين`
- char_accuracy: 0.455
- CER contribution edit/gold_len: 30/55

### hf_quran_md_ayah_route_minshawy_mujawwad_006_041_024872
- gold: `بل اياه تدعون فيكشف ما تدعون اليه ان شاء وتنسون ما تشركون`
- pred: `بالي ياتدعون فيكشف ما تدعون الي انشاء`
- char_accuracy: 0.609
- CER contribution edit/gold_len: 18/46

### hf_quran_md_ayah_route_husary_mujawwad_012_089_050524
- gold: `قال هل علمتم ما فعلتم بيوسف واخيه اذ انتم جاهلون`
- pred: `قال هالعلمتم ما فعلتم بيوسف واخي`
- char_accuracy: 0.641
- CER contribution edit/gold_len: 14/39

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_003_158_013501
- gold: `ولئن متم او قتلتم لالي الله تحشرون`
- pred: `ولئم متتم او قتلتم لالله تحشعون`
- char_accuracy: 0.786
- CER contribution edit/gold_len: 6/28

### hf_quran_md_ayah_route_banna_018_036_065255
- gold: `وما اظن الساعة قائمة ولئن رددت الي ربي لاجدن خيرا منها منقلبا`
- pred: `وما اظن الساعة قائمة ولاردت اي ربي لاددا خير من اام قلب`
- char_accuracy: 0.800
- CER contribution edit/gold_len: 10/50

### hf_quran_md_ayah_route_banna_018_041_065405
- gold: `او يصبح ماؤها غورا فلن تستطيع له طلبا`
- pred: `اولصب حما اهي غورا فلا ان تستطيع له طلبا`
- char_accuracy: 0.833
- CER contribution edit/gold_len: 5/30

### hf_quran_md_ayah_route_husary_mujawwad_012_079_050224
- gold: `قال معاذ الله ان ناخذ الا من وجدنا متاعنا عنده انا اذا لظالمون`
- pred: `قال معاذ الله ان اخذ الا من وجدنا متاعنا عنده ان اذل لوانا`
- char_accuracy: 0.840
- CER contribution edit/gold_len: 8/50

### hf_quran_md_ayah_route_minshawy_mujawwad_007_109_031862
- gold: `قال الملا من قوم فرعون ان هذا لساحر عليم`
- pred: `قال الملوم قوم في العون ان هذا لساحر عليم`
- char_accuracy: 0.844
- CER contribution edit/gold_len: 5/32

### hf_quran_md_ayah_route_husary_mujawwad_015_045_055384
- gold: `ان المتقين في جنات وعيون`
- pred: `ان المتكين في جناتهم عيون`
- char_accuracy: 0.850
- CER contribution edit/gold_len: 3/20

### hf_quran_md_ayah_route_minshawy_mujawwad_007_025_029342
- gold: `قال فيها تحيون وفيها تموتون ومنها تخرجون`
- pred: `قال في اات يون وفي اات موتون ومن اات اخرجون`
- char_accuracy: 0.853
- CER contribution edit/gold_len: 5/34

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_117_018271
- gold: `ان يدعون من دونه الا اناثا وان يدعون الا شيطانا مريدا`
- pred: `ين يدعون من دونه الا اناسو وين يدعون الا شيطان مريد`
- char_accuracy: 0.860
- CER contribution edit/gold_len: 6/43

### hf_quran_md_ayah_route_saood_ash_shuraym_003_082_011220
- gold: `فمن تولي بعد ذلك فاولئك هم الفاسقون`
- pred: `فمن تولي بعد ذلك فالا اكهم الفازقون`
- char_accuracy: 0.862
- CER contribution edit/gold_len: 4/29

### hf_quran_md_ayah_route_abdullah_basfar_010_008_041133
- gold: `اولئك ماواهم النار بما كانوا يكسبون`
- pred: `ولئك مؤوه النار بما كانوا يكسبون`
- char_accuracy: 0.867
- CER contribution edit/gold_len: 4/30

### hf_quran_md_ayah_route_minshawy_mujawwad_007_132_032552
- gold: `وقالوا مهما تاتنا به من اية لتسحرنا بها فما نحن لك بمؤمنين`
- pred: `وقالوا مهما تاتنا بي من ايات لتسحرنا بذا فما نحن لك بمؤمنه`
- char_accuracy: 0.872
- CER contribution edit/gold_len: 6/47

### hf_quran_md_ayah_route_minshawy_mujawwad_007_015_029042
- gold: `قال انك من المنظرين`
- pred: `قال انك من انظرين`
- char_accuracy: 0.875
- CER contribution edit/gold_len: 2/16

### hf_quran_md_ayah_route_husary_mujawwad_012_105_051004
- gold: `وكاين من اية في السماوات والارض يمرون عليها وهم عنها معرضون`
- pred: `وكائم من ايات في السماوات والارض يمرون عليها وهم عنها معرين`
- char_accuracy: 0.878
- CER contribution edit/gold_len: 6/49

### hf_quran_md_ayah_route_minshawy_mujawwad_007_204_034712
- gold: `واذا قرئ القران فاستمعوا له وانصتوا لعلكم ترحمون`
- pred: `واذا قرئ القران فاستمع له وانصت لعلكم ترحون`
- char_accuracy: 0.878
- CER contribution edit/gold_len: 5/41

### hf_quran_md_ayah_route_saood_ash_shuraym_002_042_001440
- gold: `ولا تلبسوا الحق بالباطل وتكتموا الحق وانتم تعلمون`
- pred: `ولا تلبس الحق بالباطن وتكتم الحق وانتم تعلمون`
- char_accuracy: 0.881
- CER contribution edit/gold_len: 5/42

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_167_019771
- gold: `ان الذين كفروا وصدوا عن سبيل الله قد ضلوا ضلالا بعيدا`
- pred: `ان الذين كغرو وصدوا عن سبيل الله قد ظلوا ضلالا بيدي`
- char_accuracy: 0.884
- CER contribution edit/gold_len: 5/43

### hf_quran_md_ayah_route_banna_017_056_062525
- gold: `قل ادعوا الذين زعمتم من دونه فلا يملكون كشف الضر عنكم ولا تحويلا`
- pred: `قلذ الذين زعمتم من دونه فلا يملكون كشف الضرعاكم ولا تحويلا`
- char_accuracy: 0.885
- CER contribution edit/gold_len: 6/52

### hf_quran_md_ayah_route_minshawy_mujawwad_007_139_032762
- gold: `ان هؤلاء متبر ما هم فيه وباطل ما كانوا يعملون`
- pred: `انا انا امتبر ماهم فيه وباطل ما كانوا يعملون`
- char_accuracy: 0.889
- CER contribution edit/gold_len: 4/36

### hf_quran_md_ayah_route_minshawy_mujawwad_007_174_033812
- gold: `وكذلك نفصل الايات ولعلهم يرجعون`
- pred: `وكذلك نفس الاهات ولعلهم يرجعون`
- char_accuracy: 0.889
- CER contribution edit/gold_len: 3/27

### hf_quran_md_ayah_route_banna_018_072_066335
- gold: `قال الم اقل انك لن تستطيع معي صبرا`
- pred: `قال الم اقول انك لان تستطيع معي صبر`
- char_accuracy: 0.889
- CER contribution edit/gold_len: 3/27

### hf_quran_md_ayah_route_abdullah_basfar_009_078_039363
- gold: `الم يعلموا ان الله يعلم سرهم ونجواهم وان الله علام الغيوب`
- pred: `الم يعلم ان الله يعلم سبراهم ونجواهم وان الله علام الغيوم`
- char_accuracy: 0.894
- CER contribution edit/gold_len: 5/47

### hf_quran_md_ayah_route_abdullah_basfar_008_055_036423
- gold: `ان شر الدواب عند الله الذين كفروا فهم لا يؤمنون`
- pred: `ان شردوا باندي الله الذين كفروا فهم لا يؤمنون`
- char_accuracy: 0.895
- CER contribution edit/gold_len: 4/38

### hf_quran_md_ayah_route_husary_mujawwad_015_056_055714
- gold: `قال ومن يقنط من رحمة ربه الا الضالون`
- pred: `قال ومن يقنط من رحمة ربه الا ضاللون`
- char_accuracy: 0.897
- CER contribution edit/gold_len: 3/29

### hf_quran_md_ayah_route_saood_ash_shuraym_002_179_005550
- gold: `ولكم في القصاص حياة يا اولي الالباب لعلكم تتقون`
- pred: `ولكم في القصاص حياتي ياوني الاباب لعلكم تتقون`
- char_accuracy: 0.897
- CER contribution edit/gold_len: 4/39


## Best examples

### hf_quran_md_ayah_route_saood_ash_shuraym_002_066_002160
- gold: `فجعلناها نكالا لما بين يديها وما خلفها وموعظة للمتقين`
- pred: `فجعلناها نكالا لما بين يديها وما خلفها وموعظة للمتقين`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_saood_ash_shuraym_002_077_002490
- gold: `اولا يعلمون ان الله يعلم ما يسرون وما يعلنون`
- pred: `اولا يعلمون ان الله يعلم ما يسرون وما يعلنون`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_saood_ash_shuraym_003_089_011430
- gold: `الا الذين تابوا من بعد ذلك واصلحوا فان الله غفور رحيم`
- pred: `الا الذين تابوا من بعد ذلك واصلحوا فان الله غفور رحيم`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_068_016801
- gold: `ولهديناهم صراطا مستقيما`
- pred: `ولهديناهم صراطا مستقيما`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_005_030_020941
- gold: `فطوعت له نفسه قتل اخيه فقتله فاصبح من الخاسرين`
- pred: `فطوعت له نفسه قتل اخيه فقتله فاصبح من الخاسرين`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdullah_basfar_010_086_043473
- gold: `ونجنا برحمتك من القوم الكافرين`
- pred: `ونجنا برحمتك من القوم الكافرين`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_abdullah_basfar_011_079_046533
- gold: `قالوا لقد علمت ما لنا في بناتك من حق وانك لتعلم ما نريد`
- pred: `قالوا لقد علمت ما لنا في بناتك من حق وانك لتعلم ما نريد`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_husary_mujawwad_015_069_056104
- gold: `واتقوا الله ولا تخزون`
- pred: `واتقوا الله ولا تخزون`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_husary_mujawwad_016_084_059524
- gold: `ويوم نبعث من كل امة شهيدا ثم لا يؤذن للذين كفروا ولا هم يستعتبون`
- pred: `ويوم نبعث من كل امة شهيدا ثم لا يؤذن للذين كفروا ولاهم يستعتبون`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_banna_016_109_060275
- gold: `لا جرم انهم في الاخرة هم الخاسرون`
- pred: `لا جرم انهم في الاخرة هم الخاسرون`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_banna_018_100_067175
- gold: `وعرضنا جهنم يومئذ للكافرين عرضا`
- pred: `وعرضنا جهنم يومئذ للكافرين عرضا`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_banna_019_009_067745
- gold: `قال كذلك قال ربك هو علي هين وقد خلقتك من قبل ولم تك شيئا`
- pred: `قالك ذلك قال ربك هو علي هين وقد خلقتك من قبل ولم تك شيئا`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_banna_019_031_068405
- gold: `وجعلني مباركا اين ما كنت واوصاني بالصلاة والزكاة ما دمت حيا`
- pred: `وجعلني مباركا اينما كنتو اوصاني بالصلاة والزكاة ما دمتحيا`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_banna_019_057_069185
- gold: `ورفعناه مكانا عليا`
- pred: `ورفعناه مكانا عليا`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_banna_019_086_070055
- gold: `ونسوق المجرمين الي جهنم وردا`
- pred: `ونسوق المجرمين الي جهن موردا`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_banna_020_034_071435
- gold: `ونذكرك كثيرا`
- pred: `ونذكرك كثيرا`
- char_accuracy: 1.000

### hf_quran_md_ayah_route_saood_ash_shuraym_003_102_011820
- gold: `يا ايها الذين امنوا اتقوا الله حق تقاته ولا تموتن الا وانتم مسلمون`
- pred: `يا ايها الذين امنوا اتقوا الله حقت قاته ولا تموت ان الا وانتم مسلمون`
- char_accuracy: 0.981

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_112_018121
- gold: `ومن يكسب خطيئة او اثما ثم يرم به بريئا فقد احتمل بهتانا واثما مبينا`
- pred: `ومن يكسب خطيئة او اثم ثم يرم به بريئا فقد احتمل بهتانا واثما مبينا`
- char_accuracy: 0.981

### hf_quran_md_ayah_route_husary_mujawwad_013_021_051814
- gold: `والذين يصلون ما امر الله به ان يوصل ويخشون ربهم ويخافون سوء الحساب`
- pred: `والذين يصلون ما امر الله به ان يوصل ويخشون ربهم ويخافون سوالحساب`
- char_accuracy: 0.981

### hf_quran_md_ayah_route_banna_018_030_065075
- gold: `ان الذين امنوا وعملوا الصالحات انا لا نضيع اجر من احسن عملا`
- pred: `ان الذين امنوا وعملوا الصالحات اننا لا نضيع اجر من احسن عملا`
- char_accuracy: 0.979
