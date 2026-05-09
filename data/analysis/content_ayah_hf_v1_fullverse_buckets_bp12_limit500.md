# Full-verse content diagnostics

- checkpoint: `checkpoints\content_ayah_hf_v1_hd96.pt`
- manifest: `data\manifests\content_v6a_short_hf_ayah_r1_hf_ayah_clean_all.jsonl`
- split: `val`
- blank_penalty: 1.2

## Overall

- samples: 448
- exact_match: 0.047
- char_accuracy: 0.742
- edit_distance: 7.038
- avg_gold_len: 27.5
- avg_pred_len: 25.8

## Buckets

| bucket | samples | exact | char_accuracy | edit_distance | avg_gold_len | avg_pred_len |
|---|---:|---:|---:|---:|---:|---:|
| 001_020 | 176 | 0.108 | 0.723 | 4.091 | 14.7 | 14.0 |
| 021_040 | 178 | 0.011 | 0.761 | 7.129 | 29.5 | 27.4 |
| 041_060 | 94 | 0.000 | 0.742 | 12.383 | 47.7 | 45.0 |

## Worst examples

### hf_quran_md_ayah_route_minshawy_mujawwad_008_028_035612
- bucket: 041_060
- gold: `واعلمواانمااموالكمواولادكمفتنةواناللهعندهاجرعظيم`
- pred: `ولهاانماممولتمموولاتكفنتوانواعدوايهنيوالموانمالاكوااملاكممفنتوانوعندعاواهمين`
- char_accuracy: 0.000
- lengths gold/pred: 48/76

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_073_013_164623
- bucket: 021_040
- gold: `وطعاماذاغصةوعذابااليما`
- pred: `نان`
- char_accuracy: 0.045
- lengths gold/pred: 22/3

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_043_162523
- bucket: 021_040
- gold: `يوميخرجونمنالاجداثسراعاكانهمالينصبيوفضون`
- pred: `عاههاهااين`
- char_accuracy: 0.150
- lengths gold/pred: 40/10

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_020_163993
- bucket: 021_040
- gold: `قلانماادعوربيولااشركبهاحدا`
- pred: `فاااا`
- char_accuracy: 0.154
- lengths gold/pred: 26/5

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_037_167623
- bucket: 001_020
- gold: `الميكنطفةمنمنييمني`
- pred: `وللننها`
- char_accuracy: 0.167
- lengths gold/pred: 18/7

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_030_167413
- bucket: 001_020
- gold: `اليربكيومئذالمساق`
- pred: `ولاكككنا`
- char_accuracy: 0.176
- lengths gold/pred: 17/8

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_040_167713
- bucket: 021_040
- gold: `اليسذلكبقادرعليانيحييالموتي`
- pred: `فيررايين`
- char_accuracy: 0.185
- lengths gold/pred: 27/8

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_016_161713
- bucket: 001_020
- gold: `نزاعةللشوي`
- pred: `وعاالسا`
- char_accuracy: 0.200
- lengths gold/pred: 10/7

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_001_163423
- bucket: 041_060
- gold: `قلاوحياليانهاستمعنفرمنالجنفقالوااناسمعناقراناعجبا`
- pred: `ينااينيلنساا`
- char_accuracy: 0.204
- lengths gold/pred: 49/12

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_012_163753
- bucket: 021_040
- gold: `واناظنناانلننعجزاللهفيالارضولننعجزههربا`
- pred: `فاعععاهاهاانا`
- char_accuracy: 0.205
- lengths gold/pred: 39/13

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_001_162583
- bucket: 041_060
- gold: `اناارسلنانوحااليقومهانانذرقومكمنقبلانياتيهمعذاباليم`
- pred: `هنساههاناياقاين`
- char_accuracy: 0.216
- lengths gold/pred: 51/15

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_004_157333
- bucket: 041_060
- gold: `ثمارجعالبصركرتينينقلباليكالبصرخاسئاوهوحسير`
- pred: `اعاهاابخاوين`
- char_accuracy: 0.238
- lengths gold/pred: 42/12

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_030_165733
- bucket: 001_020
- gold: `عليهاتسعةعشر`
- pred: `ولسر`
- char_accuracy: 0.250
- lengths gold/pred: 12/4

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_016_163033
- bucket: 021_040
- gold: `وجعلالقمرفيهننوراوجعلالشمسسراجا`
- pred: `وملهااووااسلاة`
- char_accuracy: 0.258
- lengths gold/pred: 31/14

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_014_157633
- bucket: 021_040
- gold: `الايعلممنخلقوهواللطيفالخبير`
- pred: `للععلهوواممين`
- char_accuracy: 0.259
- lengths gold/pred: 27/13

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_025_163303
- bucket: 041_060
- gold: `مماخطيئاتهماغرقوافادخلوانارافلميجدوالهممندوناللهانصارا`
- pred: `نوامعيهمييوموااااينيااا`
- char_accuracy: 0.259
- lengths gold/pred: 54/23

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_008_161473
- bucket: 001_020
- gold: `يومتكونالسماءكالمهل`
- pred: `وامهعامن`
- char_accuracy: 0.263
- lengths gold/pred: 19/8

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_020_165433
- bucket: 001_020
- gold: `ثمقتلكيفقدر`
- pred: `ونلييير`
- char_accuracy: 0.273
- lengths gold/pred: 11/7

### hf_quran_md_ayah_route_muhsin_al_qasim_096_007_183376
- bucket: 001_020
- gold: `انراهاستغني`
- pred: `والايوتان`
- char_accuracy: 0.273
- lengths gold/pred: 11/9

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_016_157693
- bucket: 021_040
- gold: `اامنتممنفيالسماءانيخسفبكمالارضفاذاهيتمور`
- pred: `كعلسنسفرفااالمون`
- char_accuracy: 0.275
- lengths gold/pred: 40/16

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_073_014_164653
- bucket: 021_040
- gold: `يومترجفالارضوالجبالوكانتالجبالكثيبامهيلا`
- pred: `وهلرااالياهيمنكا`
- char_accuracy: 0.275
- lengths gold/pred: 40/16

### hf_quran_md_ayah_route_minshawy_mujawwad_006_041_024872
- bucket: 041_060
- gold: `بلاياهتدعونفيكشفماتدعوناليهانشاءوتنسونماتشركون`
- pred: `بذيتترنفيسكماادانالهيشيشكماتدنالهساافااسممااةشركون`
- char_accuracy: 0.283
- lengths gold/pred: 46/50

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_022_167173
- bucket: 001_020
- gold: `وجوهيومئذناضرة`
- pred: `ونار`
- char_accuracy: 0.286
- lengths gold/pred: 14/4

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_011_157543
- bucket: 021_040
- gold: `فاعترفوابذنبهمفسحقالاصحابالسعير`
- pred: `فعاللهلنالللعااين`
- char_accuracy: 0.290
- lengths gold/pred: 31/17

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_021_161863
- bucket: 001_020
- gold: `واذامسهالخيرمنوعا`
- pred: `ااالسههعااا`
- char_accuracy: 0.294
- lengths gold/pred: 17/11

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_038_165973
- bucket: 001_020
- gold: `كلنفسبماكسبترهينة`
- pred: `لفسفيية`
- char_accuracy: 0.294
- lengths gold/pred: 17/7

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_031_162163
- bucket: 021_040
- gold: `فمنابتغيوراءذلكفاولئكهمالعادون`
- pred: `فنللااهالاكين`
- char_accuracy: 0.300
- lengths gold/pred: 30/13

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_068_034_159133
- bucket: 021_040
- gold: `انللمتقينعندربهمجناتالنعيم`
- pred: `نريلييهالامهمنهفمنعي`
- char_accuracy: 0.308
- lengths gold/pred: 26/20

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_026_163333
- bucket: 021_040
- gold: `وقالنوحربلاتذرعليالارضمنالكافرينديارا`
- pred: `ولاللنليهاايهييرا`
- char_accuracy: 0.324
- lengths gold/pred: 37/17

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_068_014_158533
- bucket: 001_020
- gold: `انكانذامالوبنين`
- pred: `نكاين`
- char_accuracy: 0.333
- lengths gold/pred: 15/5
