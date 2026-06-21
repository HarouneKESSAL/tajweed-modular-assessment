# Full-verse content diagnostics

- checkpoint: `checkpoints\content_ayah_hf_v1_hd96.pt`
- manifest: `data\manifests\content_v6a_short_hf_ayah_r1_hf_ayah_clean_all.jsonl`
- split: `val`
- blank_penalty: 0.4

## Overall

- samples: 448
- exact_match: 0.049
- char_accuracy: 0.729
- edit_distance: 7.438
- avg_gold_len: 27.5
- avg_pred_len: 23.6

## Buckets

| bucket | samples | exact | char_accuracy | edit_distance | avg_gold_len | avg_pred_len |
|---|---:|---:|---:|---:|---:|---:|
| 001_020 | 176 | 0.108 | 0.715 | 4.222 | 14.7 | 12.6 |
| 021_040 | 178 | 0.017 | 0.746 | 7.601 | 29.5 | 25.1 |
| 041_060 | 94 | 0.000 | 0.725 | 13.149 | 47.7 | 41.1 |

## Worst examples

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_073_013_164623
- bucket: 021_040
- gold: `وطعاماذاغصةوعذابااليما`
- pred: `ن`
- char_accuracy: 0.000
- lengths gold/pred: 22/1

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_001_163423
- bucket: 041_060
- gold: `قلاوحياليانهاستمعنفرمنالجنفقالوااناسمعناقراناعجبا`
- pred: `نيا`
- char_accuracy: 0.041
- lengths gold/pred: 49/3

### hf_quran_md_ayah_route_minshawy_mujawwad_008_028_035612
- bucket: 041_060
- gold: `واعلمواانمااموالكمواولادكمفتنةواناللهعندهاجرعظيم`
- pred: `ولهانماممولتمموولاكفنتوانوادواهنيوالموانمالاكواملاكمفنتوانوعدعاوامين`
- char_accuracy: 0.042
- lengths gold/pred: 48/68

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_012_163753
- bucket: 021_040
- gold: `واناظنناانلننعجزاللهفيالارضولننعجزههربا`
- pred: `ها`
- char_accuracy: 0.051
- lengths gold/pred: 39/2

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_014_157633
- bucket: 021_040
- gold: `الايعلممنخلقوهواللطيفالخبير`
- pred: `ومين`
- char_accuracy: 0.074
- lengths gold/pred: 27/4

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_016_157693
- bucket: 021_040
- gold: `اامنتممنفيالسماءانيخسفبكمالارضفاذاهيتمور`
- pred: `عرفون`
- char_accuracy: 0.075
- lengths gold/pred: 40/5

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_020_163993
- bucket: 021_040
- gold: `قلانماادعوربيولااشركبهاحدا`
- pred: `فاا`
- char_accuracy: 0.077
- lengths gold/pred: 26/3

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_035_167563
- bucket: 001_020
- gold: `ثماوليلكفاولي`
- pred: `ي`
- char_accuracy: 0.077
- lengths gold/pred: 13/1

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_030_165733
- bucket: 001_020
- gold: `عليهاتسعةعشر`
- pred: `ر`
- char_accuracy: 0.083
- lengths gold/pred: 12/1

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_064_005_156103
- bucket: 041_060
- gold: `المياتكمنباالذينكفروامنقبلفذاقواوبالامرهمولهمعذاباليم`
- pred: `نلعلين`
- char_accuracy: 0.094
- lengths gold/pred: 53/6

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_001_162583
- bucket: 041_060
- gold: `اناارسلنانوحااليقومهانانذرقومكمنقبلانياتيهمعذاباليم`
- pred: `ناهنين`
- char_accuracy: 0.098
- lengths gold/pred: 51/6

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_031_162163
- bucket: 021_040
- gold: `فمنابتغيوراءذلكفاولئكهمالعادون`
- pred: `فهين`
- char_accuracy: 0.100
- lengths gold/pred: 30/4

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_008_161473
- bucket: 001_020
- gold: `يومتكونالسماءكالمهل`
- pred: `امن`
- char_accuracy: 0.105
- lengths gold/pred: 19/3

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_040_167713
- bucket: 021_040
- gold: `اليسذلكبقادرعليانيحييالموتي`
- pred: `فيرن`
- char_accuracy: 0.111
- lengths gold/pred: 27/4

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_030_167413
- bucket: 001_020
- gold: `اليربكيومئذالمساق`
- pred: `وكا`
- char_accuracy: 0.118
- lengths gold/pred: 17/3

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_004_157333
- bucket: 041_060
- gold: `ثمارجعالبصركرتينينقلباليكالبصرخاسئاوهوحسير`
- pred: `اابين`
- char_accuracy: 0.119
- lengths gold/pred: 42/5

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_043_162523
- bucket: 021_040
- gold: `يوميخرجونمنالاجداثسراعاكانهمالينصبيوفضون`
- pred: `عههاين`
- char_accuracy: 0.125
- lengths gold/pred: 40/6

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_073_014_164653
- bucket: 021_040
- gold: `يومترجفالارضوالجبالوكانتالجبالكثيبامهيلا`
- pred: `وهانكا`
- char_accuracy: 0.125
- lengths gold/pred: 40/6

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_011_157543
- bucket: 021_040
- gold: `فاعترفوابذنبهمفسحقالاصحابالسعير`
- pred: `فلهلين`
- char_accuracy: 0.129
- lengths gold/pred: 31/6

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_025_163303
- bucket: 041_060
- gold: `مماخطيئاتهماغرقوافادخلوانارافلميجدوالهممندوناللهانصارا`
- pred: `وممواايا`
- char_accuracy: 0.130
- lengths gold/pred: 54/8

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_026_163333
- bucket: 021_040
- gold: `وقالنوحربلاتذرعليالارضمنالكافرينديارا`
- pred: `وانرا`
- char_accuracy: 0.135
- lengths gold/pred: 37/5

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_022_167173
- bucket: 001_020
- gold: `وجوهيومئذناضرة`
- pred: `ور`
- char_accuracy: 0.143
- lengths gold/pred: 14/2

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_037_167623
- bucket: 001_020
- gold: `الميكنطفةمنمنييمني`
- pred: `ولننها`
- char_accuracy: 0.167
- lengths gold/pred: 18/6

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_038_165973
- bucket: 001_020
- gold: `كلنفسبماكسبترهينة`
- pred: `سفية`
- char_accuracy: 0.176
- lengths gold/pred: 17/4

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_002_162613
- bucket: 021_040
- gold: `قالياقومانيلكمنذيرمبين`
- pred: `مكين`
- char_accuracy: 0.182
- lengths gold/pred: 22/4

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_016_163033
- bucket: 021_040
- gold: `وجعلالقمرفيهننوراوجعلالشمسسراجا`
- pred: `وهاواسلة`
- char_accuracy: 0.194
- lengths gold/pred: 31/8

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_016_161713
- bucket: 001_020
- gold: `نزاعةللشوي`
- pred: `والا`
- char_accuracy: 0.200
- lengths gold/pred: 10/4

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_009_166783
- bucket: 001_020
- gold: `وجمعالشمسوالقمر`
- pred: `وار`
- char_accuracy: 0.200
- lengths gold/pred: 15/3

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_068_034_159133
- bucket: 021_040
- gold: `انللمتقينعندربهمجناتالنعيم`
- pred: `نيلههنهني`
- char_accuracy: 0.231
- lengths gold/pred: 26/9

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_069_049_161143
- bucket: 021_040
- gold: `وانالنعلمانمنكممكذبين`
- pred: `ننينجين`
- char_accuracy: 0.238
- lengths gold/pred: 21/7
