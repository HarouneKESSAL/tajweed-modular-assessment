# Expected ayah CTC scoring

- manifest: `data\manifests\content_v6a_short_hf_ayah_r1_hf_ayah_clean_all.jsonl`
- split: `val`
- checkpoint: `checkpoints\content_ayah_hf_v2_balanced_hd96.pt`
- decoder_config: `configs\content_ayah_decoder_bp12.json`
- blank_penalty: 1.2

## Overall

- samples: `448`
- free_exact_rate: `0.0625`
- expected_text_accepted_rate: `0.11383928571428571`
- expected_text_strong_review_rate: `0.40401785714285715`
- expected_text_plausible_review_rate: `0.24330357142857142`
- avg_free_char_accuracy: `0.7621132140627868`
- avg_expected_ctc_loss_per_char: `0.8407779494098053`
- avg_expected_ctc_confidence: `0.5052350407191526`
- verdict_counts: `{'accepted_free_decode_exact': 28, 'expected_text_strong_but_review_required': 181, 'expected_text_plausible_review_required': 109, 'free_decode_similarity_review_required': 13, 'not_supported': 94, 'accepted_expected_text_near_exact_review_recommended': 23}`

## By reciter

| reciter | samples | free exact | exp accepted | exp strong review | free char | exp loss/char | exp confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| ibrahim_akhdar | 33 | 0.091 | 0.273 | 0.636 | 0.871 | 0.433 | 0.661 |
| ali_jaber | 20 | 0.250 | 0.250 | 0.550 | 0.892 | 0.440 | 0.670 |
| hussary.teacher | 31 | 0.194 | 0.226 | 0.613 | 0.886 | 0.445 | 0.677 |
| alafasy | 23 | 0.130 | 0.217 | 0.652 | 0.883 | 0.456 | 0.644 |
| abdul_basit_murattal | 46 | 0.087 | 0.217 | 0.565 | 0.871 | 0.477 | 0.633 |
| ghamadi | 29 | 0.103 | 0.138 | 0.621 | 0.841 | 0.507 | 0.613 |
| abdullah_basfar | 11 | 0.000 | 0.091 | 0.636 | 0.848 | 0.530 | 0.598 |
| banna | 31 | 0.065 | 0.097 | 0.581 | 0.854 | 0.530 | 0.601 |
| saood_ash_shuraym | 13 | 0.154 | 0.154 | 0.462 | 0.851 | 0.560 | 0.592 |
| husary_mujawwad | 19 | 0.000 | 0.053 | 0.684 | 0.828 | 0.591 | 0.560 |
| warsh_yassin | 31 | 0.000 | 0.065 | 0.419 | 0.814 | 0.656 | 0.541 |
| abu_bakr_ash_shaatree | 56 | 0.000 | 0.018 | 0.179 | 0.707 | 0.978 | 0.400 |
| abdurrahmaan_as_sudais | 11 | 0.000 | 0.000 | 0.091 | 0.724 | 0.983 | 0.398 |
| muhsin_al_qasim | 36 | 0.000 | 0.028 | 0.056 | 0.660 | 1.175 | 0.340 |
| minshawy_mujawwad | 12 | 0.000 | 0.000 | 0.083 | 0.637 | 1.218 | 0.345 |
| warsh_husary | 5 | 0.000 | 0.000 | 0.000 | 0.547 | 1.572 | 0.255 |
| abdullaah_3awwaad_al_juhaynee | 41 | 0.000 | 0.000 | 0.000 | 0.319 | 2.468 | 0.090 |

## Best expected-text support

### hf_quran_md_ayah_route_ali_jaber_029_001_100208
- reciter: ali_jaber
- verdict: accepted_free_decode_exact
- free_score: 100.0
- free_exact: True
- expected_loss_per_char: 0.04498085379600525
- expected_confidence: 0.9560157857311301
- gold: `الم`
- free_prediction: `الم`

### hf_quran_md_ayah_route_ali_jaber_032_001_105098
- reciter: ali_jaber
- verdict: accepted_free_decode_exact
- free_score: 100.0
- free_exact: True
- expected_loss_per_char: 0.05602757136027018
- expected_confidence: 0.9455130664496177
- gold: `الم`
- free_prediction: `الم`

### hf_quran_md_ayah_route_ali_jaber_030_001_102278
- reciter: ali_jaber
- verdict: accepted_free_decode_exact
- free_score: 100.0
- free_exact: True
- expected_loss_per_char: 0.05662267406781515
- expected_confidence: 0.9449505564559655
- gold: `الم`
- free_prediction: `الم`

### hf_quran_md_ayah_route_ali_jaber_031_001_104078
- reciter: ali_jaber
- verdict: accepted_free_decode_exact
- free_score: 100.0
- free_exact: True
- expected_loss_per_char: 0.057483136653900146
- expected_confidence: 0.9441378115748621
- gold: `الم`
- free_prediction: `الم`

### hf_quran_md_ayah_route_abdul_basit_murattal_026_001_087967
- reciter: abdul_basit_murattal
- verdict: accepted_free_decode_exact
- free_score: 100.0
- free_exact: True
- expected_loss_per_char: 0.09674604733784993
- expected_confidence: 0.9077865116645883
- gold: `طسم`
- free_prediction: `طسم`

### hf_quran_md_ayah_route_hussary.teacher_036_001_111159
- reciter: hussary.teacher
- verdict: accepted_free_decode_exact
- free_score: 100.0
- free_exact: True
- expected_loss_per_char: 0.10885671526193619
- expected_confidence: 0.8968589144868294
- gold: `يس`
- free_prediction: `يس`

### hf_quran_md_ayah_route_hussary.teacher_036_063_113019
- reciter: hussary.teacher
- verdict: accepted_free_decode_exact
- free_score: 100.0
- free_exact: True
- expected_loss_per_char: 0.11598710786728632
- expected_confidence: 0.8904867034962316
- gold: `هذهجهنمالتيكنتمتوعدون`
- free_prediction: `هذهجهنمالتيكنتمتوعدون`

### hf_quran_md_ayah_route_warsh_yassin_021_110_077766
- reciter: warsh_yassin
- verdict: accepted_expected_text_near_exact_review_recommended
- free_score: 96.88
- free_exact: False
- expected_loss_per_char: 0.12281130999326706
- expected_confidence: 0.884430530019879
- gold: `انهيعلمالجهرمنالقولويعلمماتكتمون`
- free_prediction: `انهيعلموالجهرمنالقولويعلمماتكتمون`

### hf_quran_md_ayah_route_hussary.teacher_037_096_116499
- reciter: hussary.teacher
- verdict: accepted_free_decode_exact
- free_score: 100.0
- free_exact: True
- expected_loss_per_char: 0.12549060269405968
- expected_confidence: 0.8820640534138546
- gold: `واللهخلقكموماتعملون`
- free_prediction: `واللهخلقكموماتعملون`

### hf_quran_md_ayah_route_saood_ash_shuraym_002_001_000210
- reciter: saood_ash_shuraym
- verdict: accepted_free_decode_exact
- free_score: 100.0
- free_exact: True
- expected_loss_per_char: 0.128544549147288
- expected_confidence: 0.8793743861667375
- gold: `الم`
- free_prediction: `الم`

### hf_quran_md_ayah_route_saood_ash_shuraym_003_001_008790
- reciter: saood_ash_shuraym
- verdict: accepted_free_decode_exact
- free_score: 100.0
- free_exact: True
- expected_loss_per_char: 0.13169389963150024
- expected_confidence: 0.8766092844405399
- gold: `الم`
- free_prediction: `الم`

### hf_quran_md_ayah_route_abdul_basit_murattal_026_218_094477
- reciter: abdul_basit_murattal
- verdict: accepted_free_decode_exact
- free_score: 100.0
- free_exact: True
- expected_loss_per_char: 0.1488235314687093
- expected_confidence: 0.8617211681511824
- gold: `الذييراكحينتقوم`
- free_prediction: `الذييراكحينتقوم`

### hf_quran_md_ayah_route_abdul_basit_murattal_026_225_094687
- reciter: abdul_basit_murattal
- verdict: accepted_expected_text_near_exact_review_recommended
- free_score: 95.45
- free_exact: False
- expected_loss_per_char: 0.15020851655439896
- expected_confidence: 0.8605285232735965
- gold: `المترانهمفيكلواديهيمون`
- free_prediction: `المتراانهمفيكلواديهيمون`

### hf_quran_md_ayah_route_ghamadi_053_043_144792
- reciter: ghamadi
- verdict: accepted_free_decode_exact
- free_score: 100.0
- free_exact: True
- expected_loss_per_char: 0.15121906598409016
- expected_confidence: 0.8596593559074294
- gold: `وانههواضحكوابكي`
- free_prediction: `وانههواضحكوابكي`

### hf_quran_md_ayah_route_hussary.teacher_037_159_118389
- reciter: hussary.teacher
- verdict: accepted_free_decode_exact
- free_score: 100.0
- free_exact: True
- expected_loss_per_char: 0.15668817127452178
- expected_confidence: 0.854970621653822
- gold: `سبحاناللهعمايصفون`
- free_prediction: `سبحاناللهعمايصفون`

### hf_quran_md_ayah_route_ali_jaber_028_001_097568
- reciter: ali_jaber
- verdict: accepted_free_decode_exact
- free_score: 100.0
- free_exact: True
- expected_loss_per_char: 0.1642918586730957
- expected_confidence: 0.8484943453049366
- gold: `طسم`
- free_prediction: `طسم`

### hf_quran_md_ayah_route_abdul_basit_murattal_026_117_091447
- reciter: abdul_basit_murattal
- verdict: accepted_free_decode_exact
- free_score: 100.0
- free_exact: True
- expected_loss_per_char: 0.1666487753391266
- expected_confidence: 0.8464968697078908
- gold: `قالربانقوميكذبون`
- free_prediction: `قالربانقوميكذبون`

### hf_quran_md_ayah_route_hussary.teacher_037_019_114189
- reciter: hussary.teacher
- verdict: accepted_free_decode_exact
- free_score: 100.0
- free_exact: True
- expected_loss_per_char: 0.16762472902025496
- expected_confidence: 0.8456711309787539
- gold: `فانماهيزجرةواحدةفاذاهمينظرون`
- free_prediction: `فانماهيزجرةواحدةفاذاهمينظرون`

### hf_quran_md_ayah_route_alafasy_043_003_129820
- reciter: alafasy
- verdict: accepted_expected_text_near_exact_review_recommended
- free_score: 96.67
- free_exact: False
- expected_loss_per_char: 0.1755753517150879
- expected_confidence: 0.8389741766484814
- gold: `اناجعلناهقراناعربيالعلكمتعقلون`
- free_prediction: `اناجعلناهقرانانعربيالعلكمتعقلون`

### hf_quran_md_ayah_route_ibrahim_akhdar_048_001_137501
- reciter: ibrahim_akhdar
- verdict: accepted_expected_text_near_exact_review_recommended
- free_score: 94.74
- free_exact: False
- expected_loss_per_char: 0.18606150777716385
- expected_confidence: 0.8302225282691836
- gold: `انافتحنالكفتحامبينا`
- free_prediction: `انافتحنالكفتحمبينا`


## Worst expected-text support

### hf_quran_md_ayah_route_warsh_husary_084_001_176534
- reciter: warsh_husary
- verdict: not_supported
- free_score: 28.57
- expected_loss_per_char: 3.0328448159354076
- expected_confidence: 0.04817838435222483
- gold: `اذاالسماءانشقت`
- free_prediction: `ابتمءشط`

### hf_quran_md_ayah_route_minshawy_mujawwad_008_028_035612
- reciter: minshawy_mujawwad
- verdict: not_supported
- free_score: 2.08
- expected_loss_per_char: 2.9865566889444985
- expected_confidence: 0.050460890466632644
- gold: `واعلمواانمااموالكمواولادكمفتنةواناللهعندهاجرعظيم`
- free_prediction: `ولنواانمامنتممولاتكفنتنواعدواجينغيوالموامالواكواملاكمفنتوانلعدعروامين`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_004_157333
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 23.81
- expected_loss_per_char: 2.982733953566778
- expected_confidence: 0.050654158268285364
- gold: `ثمارجعالبصركرتينينقلباليكالبصرخاسئاوهوحسير`
- free_prediction: `اعجاااابسصاروامين`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_012_163753
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 17.95
- expected_loss_per_char: 2.92169189453125
- expected_confidence: 0.053842514339028334
- gold: `واناظنناانلننعجزاللهفيالارضولننعجزههربا`
- free_prediction: `فاععانمباننا`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_016_161713
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 20.0
- expected_loss_per_char: 2.9202091217041017
- expected_confidence: 0.05392240977498804
- gold: `نزاعةللشوي`
- free_prediction: `وعمالسن`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_043_162523
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 22.5
- expected_loss_per_char: 2.912379264831543
- expected_confidence: 0.05434627174958583
- gold: `يوميخرجونمنالاجداثسراعاكانهمالينصبيوفضون`
- free_prediction: `عانلامهااااين`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_073_014_164653
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 27.5
- expected_loss_per_char: 2.8546422958374023
- expected_confidence: 0.05757641276032473
- gold: `يومترجفالارضوالجبالوكانتالجبالكثيبامهيلا`
- free_prediction: `والراانانهيمنمجا`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_025_163303
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 20.37
- expected_loss_per_char: 2.8479139539930554
- expected_confidence: 0.05796511273430888
- gold: `مماخطيئاتهماغرقوافادخلوانارافلميجدوالهممندوناللهانصارا`
- free_prediction: `نمعامييجيكيوموااايياا`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_030_165733
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 25.0
- expected_loss_per_char: 2.803473154703776
- expected_confidence: 0.060599226215847554
- gold: `عليهاتسعةعشر`
- free_prediction: `وللسر`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_073_013_164623
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 27.27
- expected_loss_per_char: 2.7691579298539595
- expected_confidence: 0.06271479276812869
- gold: `وطعاماذاغصةوعذابااليما`
- free_prediction: `وااااا`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_026_163333
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 35.14
- expected_loss_per_char: 2.745528556205131
- expected_confidence: 0.06421435107936607
- gold: `وقالنوحربلاتذرعليالارضمنالكافرينديارا`
- free_prediction: `واللنللاااايهتيرا`

### hf_quran_md_ayah_route_muhsin_al_qasim_096_007_183376
- reciter: muhsin_al_qasim
- verdict: not_supported
- free_score: 27.27
- expected_loss_per_char: 2.7310577739368784
- expected_confidence: 0.06515033887693879
- gold: `انراهاستغني`
- free_prediction: `ولوااواتالي`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_008_161473
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 26.32
- expected_loss_per_char: 2.717534717760588
- expected_confidence: 0.066037354633866
- gold: `يومتكونالسماءكالمهل`
- free_prediction: `وامهسعامن`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_031_162163
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 26.67
- expected_loss_per_char: 2.6346417744954427
- expected_confidence: 0.07174466557482978
- gold: `فمنابتغيوراءذلكفاولئكهمالعادون`
- free_prediction: `فنااهااكموين`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_040_167713
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 22.22
- expected_loss_per_char: 2.6216699105721935
- expected_confidence: 0.07268139000971366
- gold: `اليسذلكبقادرعليانيحييالموتي`
- free_prediction: `فيررانيينن`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_064_005_156103
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 33.96
- expected_loss_per_char: 2.6092716432967276
- expected_confidence: 0.07358812265083138
- gold: `المياتكمنباالذينكفروامنقبلفذاقواوبالامرهمولهمعذاباليم`
- free_prediction: `ثلياامنننكاامعواواااونانعلالبجين`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_001_162583
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 29.41
- expected_loss_per_char: 2.5971341600605085
- expected_confidence: 0.07448673969868523
- gold: `اناارسلنانوحااليقومهانانذرقومكمنقبلانياتيهمعذاباليم`
- free_prediction: `نساااهنانااترمجياين`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_016_163033
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 35.48
- expected_loss_per_char: 2.5651732413999495
- expected_confidence: 0.07690585700936353
- gold: `وجعلالقمرفيهننوراوجعلالشمسسراجا`
- free_prediction: `والدباووابسسسراا`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_001_163423
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 14.29
- expected_loss_per_char: 2.5614701874402104
- expected_confidence: 0.07719117148911245
- gold: `قلاوحياليانهاستمعنفرمنالجنفقالوااناسمعناقراناعجبا`
- free_prediction: `نساينننسسلا`

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_011_157543
- reciter: abdullaah_3awwaad_al_juhaynee
- verdict: not_supported
- free_score: 29.03
- expected_loss_per_char: 2.5575421241021927
- expected_confidence: 0.07749497959796613
- gold: `فاعترفوابذنبهمفسحقالاصحابالسعير`
- free_prediction: `فعلهلنسسساالسبعااين`
