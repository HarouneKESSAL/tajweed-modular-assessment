# Full-verse content diagnostics

- checkpoint: `checkpoints\content_ayah_hf_v2_balanced_hd96.pt`
- manifest: `data\manifests\content_v6a_short_hf_ayah_r1_hf_ayah_clean_all.jsonl`
- split: `val`
- blank_penalty: 1.2

## Overall

- samples: 448
- exact_match: 0.062
- char_accuracy: 0.762
- edit_distance: 6.446
- avg_gold_len: 27.5
- avg_pred_len: 26.0

## Buckets

| bucket | samples | exact | char_accuracy | edit_distance | avg_gold_len | avg_pred_len |
|---|---:|---:|---:|---:|---:|---:|
| 001_020 | 176 | 0.131 | 0.738 | 3.835 | 14.7 | 14.2 |
| 021_040 | 178 | 0.028 | 0.786 | 6.404 | 29.5 | 27.6 |
| 041_060 | 94 | 0.000 | 0.761 | 11.415 | 47.7 | 45.1 |

## Worst examples

### hf_quran_md_ayah_route_minshawy_mujawwad_008_028_035612
- bucket: 041_060
- gold: `واعلمواانمااموالكمواولادكمفتنةواناللهعندهاجرعظيم`
- pred: `ولنواانمامنتممولاتكفنتنواعدواجينغيوالموامالواكواملاكمفنتوانلعدعروامين`
- char_accuracy: 0.021
- lengths gold/pred: 48/69

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_001_163423
- bucket: 041_060
- gold: `قلاوحياليانهاستمعنفرمنالجنفقالوااناسمعناقراناعجبا`
- pred: `نساينننسسلا`
- char_accuracy: 0.143
- lengths gold/pred: 49/11

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_012_163753
- bucket: 021_040
- gold: `واناظنناانلننعجزاللهفيالارضولننعجزههربا`
- pred: `فاععانمباننا`
- char_accuracy: 0.179
- lengths gold/pred: 39/12

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_016_161713
- bucket: 001_020
- gold: `نزاعةللشوي`
- pred: `وعمالسن`
- char_accuracy: 0.200
- lengths gold/pred: 10/7

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_025_163303
- bucket: 041_060
- gold: `مماخطيئاتهماغرقوافادخلوانارافلميجدوالهممندوناللهانصارا`
- pred: `نمعامييجيكيوموااايياا`
- char_accuracy: 0.204
- lengths gold/pred: 54/21

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_037_167623
- bucket: 001_020
- gold: `الميكنطفةمنمنييمني`
- pred: `ونرلننننيا`
- char_accuracy: 0.222
- lengths gold/pred: 18/10

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_040_167713
- bucket: 021_040
- gold: `اليسذلكبقادرعليانيحييالموتي`
- pred: `فيررانيينن`
- char_accuracy: 0.222
- lengths gold/pred: 27/10

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_016_157693
- bucket: 021_040
- gold: `اامنتممنفيالسماءانيخسفبكمالارضفاذاهيتمور`
- pred: `لسسسمففااامون`
- char_accuracy: 0.225
- lengths gold/pred: 40/13

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_043_162523
- bucket: 021_040
- gold: `يوميخرجونمنالاجداثسراعاكانهمالينصبيوفضون`
- pred: `عانلامهااااين`
- char_accuracy: 0.225
- lengths gold/pred: 40/13

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_030_167413
- bucket: 001_020
- gold: `اليربكيومئذالمساق`
- pred: `وييما`
- char_accuracy: 0.235
- lengths gold/pred: 17/5

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_004_157333
- bucket: 041_060
- gold: `ثمارجعالبصركرتينينقلباليكالبصرخاسئاوهوحسير`
- pred: `اعجاااابسصاروامين`
- char_accuracy: 0.238
- lengths gold/pred: 42/17

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_030_165733
- bucket: 001_020
- gold: `عليهاتسعةعشر`
- pred: `وللسر`
- char_accuracy: 0.250
- lengths gold/pred: 12/5

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_014_157633
- bucket: 021_040
- gold: `الايعلممنخلقوهواللطيفالخبير`
- pred: `لععممهمولمين`
- char_accuracy: 0.259
- lengths gold/pred: 27/12

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_008_161473
- bucket: 001_020
- gold: `يومتكونالسماءكالمهل`
- pred: `وامهسعامن`
- char_accuracy: 0.263
- lengths gold/pred: 19/9

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_031_162163
- bucket: 021_040
- gold: `فمنابتغيوراءذلكفاولئكهمالعادون`
- pred: `فنااهااكموين`
- char_accuracy: 0.267
- lengths gold/pred: 30/12

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_073_013_164623
- bucket: 021_040
- gold: `وطعاماذاغصةوعذابااليما`
- pred: `وااااا`
- char_accuracy: 0.273
- lengths gold/pred: 22/6

### hf_quran_md_ayah_route_muhsin_al_qasim_096_007_183376
- bucket: 001_020
- gold: `انراهاستغني`
- pred: `ولوااواتالي`
- char_accuracy: 0.273
- lengths gold/pred: 11/11

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_073_014_164653
- bucket: 021_040
- gold: `يومترجفالارضوالجبالوكانتالجبالكثيبامهيلا`
- pred: `والراانانهيمنمجا`
- char_accuracy: 0.275
- lengths gold/pred: 40/16

### hf_quran_md_ayah_route_warsh_husary_084_001_176534
- bucket: 001_020
- gold: `اذاالسماءانشقت`
- pred: `ابتمءشط`
- char_accuracy: 0.286
- lengths gold/pred: 14/7

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_011_157543
- bucket: 021_040
- gold: `فاعترفوابذنبهمفسحقالاصحابالسعير`
- pred: `فعلهلنسسساالسبعااين`
- char_accuracy: 0.290
- lengths gold/pred: 31/19

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_021_161863
- bucket: 001_020
- gold: `واذامسهالخيرمنوعا`
- pred: `االسبواا`
- char_accuracy: 0.294
- lengths gold/pred: 17/8

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_001_162583
- bucket: 041_060
- gold: `اناارسلنانوحااليقومهانانذرقومكمنقبلانياتيهمعذاباليم`
- pred: `نساااهنانااترمجياين`
- char_accuracy: 0.294
- lengths gold/pred: 51/19

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_069_005_159823
- bucket: 021_040
- gold: `فاماثمودفاهلكوابالطاغية`
- pred: `االامانبمرلية`
- char_accuracy: 0.304
- lengths gold/pred: 23/13

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_020_163993
- bucket: 021_040
- gold: `قلانماادعوربيولااشركبهاحدا`
- pred: `فاااااهاا`
- char_accuracy: 0.308
- lengths gold/pred: 26/9

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_035_167563
- bucket: 001_020
- gold: `ثماوليلكفاولي`
- pred: `ولففيفيي`
- char_accuracy: 0.308
- lengths gold/pred: 13/8

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_015_165283
- bucket: 001_020
- gold: `ثميطمعانازيد`
- pred: `ثاعاالمير`
- char_accuracy: 0.333
- lengths gold/pred: 12/9

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_009_166783
- bucket: 001_020
- gold: `وجمعالشمسوالقمر`
- pred: `وااسششصقر`
- char_accuracy: 0.333
- lengths gold/pred: 15/9

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_064_005_156103
- bucket: 041_060
- gold: `المياتكمنباالذينكفروامنقبلفذاقواوبالامرهمولهمعذاباليم`
- pred: `ثلياامنننكاامعواواااونانعلالبجين`
- char_accuracy: 0.340
- lengths gold/pred: 53/32

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_068_034_159133
- bucket: 021_040
- gold: `انللمتقينعندربهمجناتالنعيم`
- pred: `انيملييولمهمنهفني`
- char_accuracy: 0.346
- lengths gold/pred: 26/17

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_026_163333
- bucket: 021_040
- gold: `وقالنوحربلاتذرعليالارضمنالكافرينديارا`
- pred: `واللنللاااايهتيرا`
- char_accuracy: 0.351
- lengths gold/pred: 37/17
