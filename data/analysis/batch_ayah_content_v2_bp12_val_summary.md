# Batch ayah content scoring

- manifest: `data\manifests\content_v6a_short_hf_ayah_r1_hf_ayah_clean_all.jsonl`
- split: `val`
- checkpoint: `checkpoints\content_ayah_hf_v2_balanced_hd96.pt`
- decoder_config: `configs\content_ayah_decoder_bp12.json`
- blank_penalty: 1.2

## Overall strict acceptance

- samples: 448
- avg_score: 76.21
- avg_char_accuracy: 0.762
- avg_edit_distance: 6.446
- exact_rate: 0.062
- accepted_rate: 0.062
- acceptance_counts: `{'accepted_exact': 28, 'not_accepted': 420}`
- quality_counts: `{'content_verified_exact': 28, 'likely_same_ayah_but_not_clean': 127, 'same_ayah_candidate_review_required': 170, 'partial_content_match_review_required': 56, 'weak_or_wrong_content': 51, 'almost_correct_review_required': 16}`

## By reciter

| reciter | samples | avg_score | char_acc | edit | exact_rate | accepted_rate |
|---|---:|---:|---:|---:|---:|---:|
| abdullaah_3awwaad_al_juhaynee | 41 | 31.93 | 0.319 | 17.561 | 0.000 | 0.000 |
| warsh_husary | 5 | 54.65 | 0.547 | 6.800 | 0.000 | 0.000 |
| minshawy_mujawwad | 12 | 63.68 | 0.637 | 15.000 | 0.000 | 0.000 |
| muhsin_al_qasim | 36 | 65.98 | 0.660 | 5.472 | 0.000 | 0.000 |
| abu_bakr_ash_shaatree | 56 | 70.72 | 0.707 | 4.839 | 0.000 | 0.000 |
| abdurrahmaan_as_sudais | 11 | 72.39 | 0.724 | 11.000 | 0.000 | 0.000 |
| warsh_yassin | 31 | 81.35 | 0.814 | 6.903 | 0.000 | 0.000 |
| husary_mujawwad | 19 | 82.80 | 0.828 | 6.632 | 0.000 | 0.000 |
| ghamadi | 29 | 84.14 | 0.841 | 3.931 | 0.103 | 0.103 |
| abdullah_basfar | 11 | 84.82 | 0.848 | 5.909 | 0.000 | 0.000 |
| saood_ash_shuraym | 13 | 85.09 | 0.851 | 6.231 | 0.154 | 0.154 |
| banna | 31 | 85.38 | 0.854 | 5.645 | 0.065 | 0.065 |
| abdul_basit_murattal | 46 | 87.10 | 0.871 | 4.304 | 0.087 | 0.087 |
| ibrahim_akhdar | 33 | 87.13 | 0.871 | 3.485 | 0.091 | 0.091 |
| alafasy | 23 | 88.34 | 0.883 | 4.043 | 0.130 | 0.130 |
| hussary.teacher | 31 | 88.57 | 0.886 | 3.387 | 0.194 | 0.194 |
| ali_jaber | 20 | 89.17 | 0.892 | 3.950 | 0.250 | 0.250 |

## Worst examples

### hf_quran_md_ayah_route_minshawy_mujawwad_008_028_035612
- reciter: minshawy_mujawwad
- score: 2.08
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `واعلمواانمااموالكمواولادكمفتنةواناللهعندهاجرعظيم`
- pred: `ولنواانمامنتممولاتكفنتنواعدواجينغيوالموامالواكواملاكمفنتوانلعدعروامين`
- edit_distance: 47

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_001_163423
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 14.29
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `قلاوحياليانهاستمعنفرمنالجنفقالوااناسمعناقراناعجبا`
- pred: `نساينننسسلا`
- edit_distance: 42

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_012_163753
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 17.95
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `واناظنناانلننعجزاللهفيالارضولننعجزههربا`
- pred: `فاععانمباننا`
- edit_distance: 32

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_016_161713
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 20.0
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `نزاعةللشوي`
- pred: `وعمالسن`
- edit_distance: 8

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_025_163303
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 20.37
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `مماخطيئاتهماغرقوافادخلوانارافلميجدوالهممندوناللهانصارا`
- pred: `نمعامييجيكيوموااايياا`
- edit_distance: 43

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_037_167623
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 22.22
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `الميكنطفةمنمنييمني`
- pred: `ونرلننننيا`
- edit_distance: 14

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_040_167713
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 22.22
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `اليسذلكبقادرعليانيحييالموتي`
- pred: `فيررانيينن`
- edit_distance: 21

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_016_157693
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 22.5
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `اامنتممنفيالسماءانيخسفبكمالارضفاذاهيتمور`
- pred: `لسسسمففااامون`
- edit_distance: 31

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_043_162523
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 22.5
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `يوميخرجونمنالاجداثسراعاكانهمالينصبيوفضون`
- pred: `عانلامهااااين`
- edit_distance: 31

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_030_167413
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 23.53
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `اليربكيومئذالمساق`
- pred: `وييما`
- edit_distance: 13

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_004_157333
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 23.81
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `ثمارجعالبصركرتينينقلباليكالبصرخاسئاوهوحسير`
- pred: `اعجاااابسصاروامين`
- edit_distance: 32

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_030_165733
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 25.0
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `عليهاتسعةعشر`
- pred: `وللسر`
- edit_distance: 9

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_014_157633
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 25.93
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `الايعلممنخلقوهواللطيفالخبير`
- pred: `لععممهمولمين`
- edit_distance: 20

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_008_161473
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 26.32
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `يومتكونالسماءكالمهل`
- pred: `وامهسعامن`
- edit_distance: 14

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_031_162163
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 26.67
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `فمنابتغيوراءذلكفاولئكهمالعادون`
- pred: `فنااهااكموين`
- edit_distance: 22

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_073_013_164623
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 27.27
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `وطعاماذاغصةوعذابااليما`
- pred: `وااااا`
- edit_distance: 16

### hf_quran_md_ayah_route_muhsin_al_qasim_096_007_183376
- reciter: muhsin_al_qasim
- score: 27.27
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `انراهاستغني`
- pred: `ولوااواتالي`
- edit_distance: 8

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_073_014_164653
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 27.5
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `يومترجفالارضوالجبالوكانتالجبالكثيبامهيلا`
- pred: `والراانانهيمنمجا`
- edit_distance: 29

### hf_quran_md_ayah_route_warsh_husary_084_001_176534
- reciter: warsh_husary
- score: 28.57
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `اذاالسماءانشقت`
- pred: `ابتمءشط`
- edit_distance: 10

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_011_157543
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 29.03
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `فاعترفوابذنبهمفسحقالاصحابالسعير`
- pred: `فعلهلنسسساالسبعااين`
- edit_distance: 22

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_021_161863
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 29.41
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `واذامسهالخيرمنوعا`
- pred: `االسبواا`
- edit_distance: 12

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_001_162583
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 29.41
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `اناارسلنانوحااليقومهانانذرقومكمنقبلانياتيهمعذاباليم`
- pred: `نساااهنانااترمجياين`
- edit_distance: 36

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_069_005_159823
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 30.43
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `فاماثمودفاهلكوابالطاغية`
- pred: `االامانبمرلية`
- edit_distance: 16

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_020_163993
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 30.77
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `قلانماادعوربيولااشركبهاحدا`
- pred: `فاااااهاا`
- edit_distance: 18

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_035_167563
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 30.77
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `ثماوليلكفاولي`
- pred: `ولففيفيي`
- edit_distance: 9

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_015_165283
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 33.33
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `ثميطمعانازيد`
- pred: `ثاعاالمير`
- edit_distance: 8

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_009_166783
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 33.33
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `وجمعالشمسوالقمر`
- pred: `وااسششصقر`
- edit_distance: 10

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_064_005_156103
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 33.96
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `المياتكمنباالذينكفروامنقبلفذاقواوبالامرهمولهمعذاباليم`
- pred: `ثلياامنننكاامعواواااونانعلالبجين`
- edit_distance: 35

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_068_034_159133
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 34.62
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `انللمتقينعندربهمجناتالنعيم`
- pred: `انيملييولمهمنهفني`
- edit_distance: 17

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_026_163333
- reciter: abdullaah_3awwaad_al_juhaynee
- score: 35.14
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `وقالنوحربلاتذرعليالارضمنالكافرينديارا`
- pred: `واللنللاااايهتيرا`
- edit_distance: 24
