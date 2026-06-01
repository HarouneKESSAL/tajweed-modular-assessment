# Whisper-small Quran ASR error buckets

## Overall

- samples: 448
- exact_compact: 0.5402
- avg_char_accuracy: 0.8884
- cer: 0.0928
- num_errors: 206

## By length

| bucket | samples | exact_compact | char_acc | CER |
|---|---:|---:|---:|---:|
| 001_020 | 176 | 0.545 | 0.846 | 0.121 |
| 021_040 | 178 | 0.579 | 0.908 | 0.097 |
| 041_060 | 94 | 0.457 | 0.931 | 0.071 |

## By reciter, worst CER first

| reciter | samples | exact_compact | char_acc | CER |
|---|---:|---:|---:|---:|
| abdullaah_3awwaad_al_juhaynee | 41 | 0.000 | 0.253 | 0.773 |
| husary_mujawwad | 19 | 0.421 | 0.927 | 0.089 |
| minshawy_mujawwad | 12 | 0.417 | 0.932 | 0.072 |
| ali_jaber | 20 | 0.500 | 0.800 | 0.043 |
| warsh_husary | 5 | 0.400 | 0.960 | 0.041 |
| saood_ash_shuraym | 13 | 0.538 | 0.854 | 0.040 |
| muhsin_al_qasim | 36 | 0.528 | 0.956 | 0.036 |
| alafasy | 23 | 0.522 | 0.935 | 0.030 |
| abu_bakr_ash_shaatree | 56 | 0.607 | 0.970 | 0.028 |
| hussary.teacher | 31 | 0.774 | 0.938 | 0.028 |
| ghamadi | 29 | 0.621 | 0.971 | 0.026 |
| abdullah_basfar | 11 | 0.364 | 0.978 | 0.024 |
| abdurrahmaan_as_sudais | 11 | 0.636 | 0.977 | 0.024 |
| warsh_yassin | 31 | 0.484 | 0.977 | 0.023 |
| banna | 31 | 0.645 | 0.982 | 0.019 |
| abdul_basit_murattal | 46 | 0.674 | 0.970 | 0.018 |
| ibrahim_akhdar | 33 | 0.788 | 0.988 | 0.013 |

## Worst examples

### hf_quran_md_ayah_route_saood_ash_shuraym_002_001_000210
- gold: `الم`
- pred: `الاخلام ميم`
- char_accuracy: 0.000
- edit/gold_len: 7/3

### hf_quran_md_ayah_route_ali_jaber_028_001_097568
- gold: `طسم`
- pred: `قاسيم ميم`
- char_accuracy: 0.000
- edit/gold_len: 6/3

### hf_quran_md_ayah_route_hussary.teacher_036_001_111159
- gold: `يس`
- pred: `ياسين`
- char_accuracy: 0.000
- edit/gold_len: 3/2

### hf_quran_md_ayah_route_alafasy_042_002_128200
- gold: `عسق`
- pred: `عن سنقاف`
- char_accuracy: 0.000
- edit/gold_len: 4/3

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_069_049_161143
- gold: `وانا لنعلم ان منكم مكذبين`
- pred: `وانا لحقهم ان يكونوا كذبين كذبين لحقهم ان يكونوا كذبين`
- char_accuracy: 0.000
- edit/gold_len: 29/21

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_015_165283
- gold: `ثم يطمع ان ازيد`
- pred: `والا اقوم بكتسين`
- char_accuracy: 0.000
- edit/gold_len: 13/12

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_050_166333
- gold: `كانهم حمر مستنفرة`
- pred: `فللن نبعه ونبعه بسرعة`
- char_accuracy: 0.000
- edit/gold_len: 15/15

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_040_167713
- gold: `اليس ذلك بقادر علي ان يحيي الموتي`
- pred: `علي السباح المتحدة والمتحدة والمتحدة والمتحدة والمتحدة والمتحدة`
- char_accuracy: 0.000
- edit/gold_len: 42/27

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_038_165973
- gold: `كل نفس بما كسبت رهينة`
- pred: `ان يوسي بحثنا بسرعة عظيم`
- char_accuracy: 0.059
- edit/gold_len: 16/17

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_008_165073
- gold: `فاذا نقر في الناقور`
- pred: `ايكذاه تقول لهم لا تقولون`
- char_accuracy: 0.062
- edit/gold_len: 15/16

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_030_167413
- gold: `الي ربك يومئذ المساق`
- pred: `وما يقولون ليكم في الساعة`
- char_accuracy: 0.118
- edit/gold_len: 15/17

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_035_167563
- gold: `ثم اولي لك فاولي`
- pred: `فانه لحول فكبر`
- char_accuracy: 0.154
- edit/gold_len: 11/13

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_031_162163
- gold: `فمن ابتغي وراء ذلك فاولئك هم العادون`
- pred: `ثم ادراك وطعاما علي ادراك ادراه امامه`
- char_accuracy: 0.167
- edit/gold_len: 25/30

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_030_165733
- gold: `عليها تسعة عشر`
- pred: `علينا المستكافرون`
- char_accuracy: 0.167
- edit/gold_len: 10/12

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_037_167623
- gold: `الم يك نطفة من مني يمني`
- pred: `عليه وطعه فانجل الاناق`
- char_accuracy: 0.167
- edit/gold_len: 15/18

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_069_005_159823
- gold: `فاما ثمود فاهلكوا بالطاغية`
- pred: `قال انا لسه ربه ربه ربه ربه`
- char_accuracy: 0.174
- edit/gold_len: 19/23

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_012_163753
- gold: `وانا ظننا ان لن نعجز الله في الارض ولن نعجزه هربا`
- pred: `وما ادراه ما قالت انه يسره علي ادراه يسره علي`
- char_accuracy: 0.179
- edit/gold_len: 32/39

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_025_163303
- gold: `مما خطيئاتهم اغرقوا فادخلوا نارا فلم يجدوا لهم من دون الله انصارا`
- pred: `انها فقل انها كانت مؤتفة بشهورها فايليهم بهم انهم كانوا مؤتفة بشهورها`
- char_accuracy: 0.185
- edit/gold_len: 44/54

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_004_157333
- gold: `ثم ارجع البصر كرتين ينقلب اليك البصر خاسئا وهو حسير`
- pred: `ثم انه لحسرت الله علي الله ومنه للمسر وصادقه ومنه سيئين`
- char_accuracy: 0.190
- edit/gold_len: 34/42

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_068_047_159523
- gold: `ام عندهم الغيب فهم يكتبون`
- pred: `والذين ادبروا بيوم القضون`
- char_accuracy: 0.190
- edit/gold_len: 17/21

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_068_034_159133
- gold: `ان للمتقين عند ربهم جنات النعيم`
- pred: `ان لن تكذب ون تبكذ بيوم احدا مليئا`
- char_accuracy: 0.192
- edit/gold_len: 21/26

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_016_163033
- gold: `وجعل القمر فيهن نورا وجعل الشمس سراجا`
- pred: `فتعايلوا ربهم في الارض ومنهم يجعلهم شرسة راجعة`
- char_accuracy: 0.194
- edit/gold_len: 25/31

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_009_166783
- gold: `وجمع الشمس والقمر`
- pred: `والذين هم من المزيد`
- char_accuracy: 0.200
- edit/gold_len: 12/15

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_075_022_167173
- gold: `وجوه يومئذ ناضرة`
- pred: `عن نميه يومئذ بلا طعام`
- char_accuracy: 0.214
- edit/gold_len: 11/14

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_073_014_164653
- gold: `يوم ترجف الارض والجبال وكانت الجبال كثيبا مهيلا`
- pred: `فتذر من ربهم جبال ويجعلهم بينات فايلا بمشاء`
- char_accuracy: 0.225
- edit/gold_len: 31/40

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_011_157543
- gold: `فاعترفوا بذنبهم فسحقا لاصحاب السعير`
- pred: `ايعتبروا منهم ان نساعد بمعفوه مساءلين`
- char_accuracy: 0.226
- edit/gold_len: 24/31

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_064_005_156103
- gold: `الم ياتكم نبا الذين كفروا من قبل فذاقوا وبال امرهم ولهم عذاب اليم`
- pred: `وانه كان لكم مهينة ربهم كما كنتم وما ربهم كما كنتم كذبين`
- char_accuracy: 0.245
- edit/gold_len: 40/53

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_014_157633
- gold: `الا يعلم من خلق وهو اللطيف الخبير`
- pred: `بلا لحق القضوة ولم يدعوه فليسل`
- char_accuracy: 0.259
- edit/gold_len: 20/27

### hf_quran_md_ayah_route_hussary.teacher_037_152_118179
- gold: `ولد الله وانهم لكاذبون`
- pred: `من افكهم ليقولون ولد الله وانهم لكاذبون`
- char_accuracy: 0.263
- edit/gold_len: 14/19

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_068_014_158533
- gold: `ان كان ذا مال وبنين`
- pred: `للن ترد رابه ومعين`
- char_accuracy: 0.267
- edit/gold_len: 11/15

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_020_165433
- gold: `ثم قتل كيف قدر`
- pred: `وانه تدريه وقدر`
- char_accuracy: 0.273
- edit/gold_len: 8/11

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_001_162583
- gold: `انا ارسلنا نوحا الي قومه ان انذر قومك من قبل ان ياتيهم عذاب اليم`
- pred: `انا سمعنا علي الانس من امامكم وتبعون رقمين يرقم علي امامكمين`
- char_accuracy: 0.294
- edit/gold_len: 36/51

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_071_026_163333
- gold: `وقال نوح رب لا تذر علي الارض من الكافرين ديارا`
- pred: `وطاله للمجرم فلنقوم بيلا جاهزين جهزا`
- char_accuracy: 0.297
- edit/gold_len: 26/37

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_067_016_157693
- gold: `اامنتم من في السماء ان يخسف بكم الارض فاذا هي تمور`
- pred: `وانه من ادبر الاخرة يسجده تطفوه لا يتعوم`
- char_accuracy: 0.300
- edit/gold_len: 28/40

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_016_161713
- gold: `نزاعة للشوي`
- pred: `والزاوة الشر`
- char_accuracy: 0.300
- edit/gold_len: 7/10

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_020_163993
- gold: `قل انما ادعو ربي ولا اشرك به احدا`
- pred: `فانما هذا هو ربنا وما كانوا لنهدفا`
- char_accuracy: 0.308
- edit/gold_len: 18/26

### hf_quran_md_ayah_route_saood_ash_shuraym_003_001_008790
- gold: `الم`
- pred: `الميم`
- char_accuracy: 0.333
- edit/gold_len: 2/3

### hf_quran_md_ayah_route_abdul_basit_murattal_026_001_087967
- gold: `طسم`
- pred: `طاسيم`
- char_accuracy: 0.333
- edit/gold_len: 2/3

### hf_quran_md_ayah_route_ali_jaber_029_001_100208
- gold: `الم`
- pred: `الميم`
- char_accuracy: 0.333
- edit/gold_len: 2/3

### hf_quran_md_ayah_route_ali_jaber_030_001_102278
- gold: `الم`
- pred: `الميم`
- char_accuracy: 0.333
- edit/gold_len: 2/3
