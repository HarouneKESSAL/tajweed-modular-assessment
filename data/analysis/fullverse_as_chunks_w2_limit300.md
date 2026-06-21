# Full-verse as chunks diagnostics

- checkpoint: `checkpoints\content_v6c_old_expanded_full_vocab_hd96.pt`
- manifest: `data\manifests\content_v6a_short_hf_ayah_r1_hf_ayah_clean_all.jsonl`
- split: `val`
- limit: 300
- max_words_per_chunk: 2
- min_window_sec: 0.35
- blank_penalty: 0.4
- window_rows: 1193

## Overall

- samples: 300
- exact_match: 0.000
- char_accuracy: 0.228
- edit_distance: 23.820
- avg_gold_len: 31.290
- avg_pred_len: 22.813
- avg_chunks: 3.977

## Buckets

| bucket | samples | exact | char_accuracy | edit_distance | avg_gold_len | avg_pred_len | avg_chunks |
|---|---:|---:|---:|---:|---:|---:|---:|
| 001_020 | 72 | 0.000 | 0.170 | 12.486 | 14.9 | 11.8 | 2.0 |
| 021_040 | 143 | 0.000 | 0.241 | 22.566 | 29.8 | 21.5 | 3.8 |
| 041_060 | 85 | 0.000 | 0.254 | 35.529 | 47.6 | 34.4 | 6.0 |

## Worst examples

### hf_quran_md_ayah_route_saood_ash_shuraym_002_001_000210
- bucket: 001_020
- gold: `الم`
- pred: `لنعيس`
- char_accuracy: 0.000
- lengths gold/pred: 3/5
- chunks: 1
  - `الم` → `لنعيس`

### hf_quran_md_ayah_route_saood_ash_shuraym_003_001_008790
- bucket: 001_020
- gold: `الم`
- pred: `ولكلحعبدس`
- char_accuracy: 0.000
- lengths gold/pred: 3/9
- chunks: 1
  - `الم` → `ولكلحعبدس`

### hf_quran_md_ayah_route_banna_020_034_071435
- bucket: 001_020
- gold: `ونذكرككثيرا`
- pred: `ملااس`
- char_accuracy: 0.000
- lengths gold/pred: 11/5
- chunks: 1
  - `ونذكرككثيرا` → `ملااس`

### hf_quran_md_ayah_route_abdul_basit_murattal_026_001_087967
- bucket: 001_020
- gold: `طسم`
- pred: `مالحين`
- char_accuracy: 0.000
- lengths gold/pred: 3/6
- chunks: 1
  - `طسم` → `مالحين`

### hf_quran_md_ayah_route_ali_jaber_028_001_097568
- bucket: 001_020
- gold: `طسم`
- pred: `اياكني`
- char_accuracy: 0.000
- lengths gold/pred: 3/6
- chunks: 1
  - `طسم` → `اياكني`

### hf_quran_md_ayah_route_ali_jaber_029_001_100208
- bucket: 001_020
- gold: `الم`
- pred: `لناس`
- char_accuracy: 0.000
- lengths gold/pred: 3/4
- chunks: 1
  - `الم` → `لناس`

### hf_quran_md_ayah_route_ali_jaber_030_001_102278
- bucket: 001_020
- gold: `الم`
- pred: `لكالناس`
- char_accuracy: 0.000
- lengths gold/pred: 3/7
- chunks: 1
  - `الم` → `لكالناس`

### hf_quran_md_ayah_route_ali_jaber_031_001_104078
- bucket: 001_020
- gold: `الم`
- pred: `لشلناس`
- char_accuracy: 0.000
- lengths gold/pred: 3/6
- chunks: 1
  - `الم` → `لشلناس`

### hf_quran_md_ayah_route_ali_jaber_032_001_105098
- bucket: 001_020
- gold: `الم`
- pred: `لكالناس`
- char_accuracy: 0.000
- lengths gold/pred: 3/7
- chunks: 1
  - `الم` → `لكالناس`

### hf_quran_md_ayah_route_hussary.teacher_036_001_111159
- bucket: 001_020
- gold: `يس`
- pred: `ملكالناس`
- char_accuracy: 0.000
- lengths gold/pred: 2/8
- chunks: 1
  - `يس` → `ملكالناس`

### hf_quran_md_ayah_route_hussary.teacher_037_179_118989
- bucket: 001_020
- gold: `وابصرفسوفيبصرون`
- pred: `ملالنعبداناس`
- char_accuracy: 0.000
- lengths gold/pred: 15/12
- chunks: 2
  - `وابصرفسوف` → `ملالنعبد`
  - `يبصرون` → `اناس`

### hf_quran_md_ayah_route_alafasy_042_002_128200
- bucket: 001_020
- gold: `عسق`
- pred: `ملكالنيك`
- char_accuracy: 0.000
- lengths gold/pred: 3/8
- chunks: 1
  - `عسق` → `ملكالنيك`

### hf_quran_md_ayah_route_ibrahim_akhdar_051_011_140561
- bucket: 001_020
- gold: `الذينهمفيغمرةساهون`
- pred: `مليانعبدمنكنستاسملرحمس`
- char_accuracy: 0.000
- lengths gold/pred: 18/22
- chunks: 3
  - `الذينهم` → `مليانعبد`
  - `فيغمرة` → `منكنستاس`
  - `ساهون` → `ملرحمس`

### hf_quran_md_ayah_route_ghamadi_056_015_149802
- bucket: 001_020
- gold: `عليسررموضونة`
- pred: `الحعيمنمنشالرنمتا`
- char_accuracy: 0.000
- lengths gold/pred: 12/17
- chunks: 2
  - `عليسرر` → `الحعيمن`
  - `موضونة` → `منشالرنمتا`

### hf_quran_md_ayah_route_ibrahim_akhdar_051_015_140681
- bucket: 001_020
- gold: `انالمتقينفيجناتوعيون`
- pred: `ملنبداسالانعبمدملحمن`
- char_accuracy: 0.050
- lengths gold/pred: 20/20
- chunks: 3
  - `انالمتقين` → `ملنبداس`
  - `فيجنات` → `الانعبمد`
  - `وعيون` → `ملحمن`

### hf_quran_md_ayah_route_saood_ash_shuraym_002_018_000720
- bucket: 001_020
- gold: `صمبكمعميفهملايرجعون`
- pred: `الرحعينالرحبدماالرحان`
- char_accuracy: 0.053
- lengths gold/pred: 19/21
- chunks: 3
  - `صمبكم` → `الرحعين`
  - `عميفهم` → `الرحبدما`
  - `لايرجعون` → `الرحان`

### hf_quran_md_ayah_route_hussary.teacher_037_163_118509
- bucket: 001_020
- gold: `الامنهوصالالجحيم`
- pred: `ملالناسالرحموالانعبوس`
- char_accuracy: 0.062
- lengths gold/pred: 16/21
- chunks: 3
  - `الامن` → `ملالناس`
  - `هوصال` → `الرحم`
  - `الجحيم` → `والانعبوس`

### hf_quran_md_ayah_route_banna_019_089_070145
- bucket: 001_020
- gold: `لقدجئتمشيئاادا`
- pred: `ملشاةمالالنباس`
- char_accuracy: 0.071
- lengths gold/pred: 14/14
- chunks: 2
  - `لقدجئتم` → `ملشاة`
  - `شيئاادا` → `مالالنباس`

### hf_quran_md_ayah_route_hussary.teacher_037_049_115089
- bucket: 001_020
- gold: `كانهنبيضمكنون`
- pred: `ملاناااس`
- char_accuracy: 0.077
- lengths gold/pred: 13/8
- chunks: 2
  - `كانهنبيض` → `ملانا`
  - `مكنون` → `ااس`

### hf_quran_md_ayah_route_husary_mujawwad_015_072_056194
- bucket: 021_040
- gold: `لعمركانهملفيسكرتهميعمهون`
- pred: `الرحماسمايانعبداسالرحيم`
- char_accuracy: 0.083
- lengths gold/pred: 24/23
- chunks: 3
  - `لعمركانهم` → `الرحماس`
  - `لفيسكرتهم` → `مايانعبداس`
  - `يعمهون` → `الرحيم`

### hf_quran_md_ayah_route_ghamadi_053_051_145032
- bucket: 001_020
- gold: `وثمودفماابقي`
- pred: `لانالعبداس`
- char_accuracy: 0.083
- lengths gold/pred: 12/10
- chunks: 2
  - `وثمودفما` → `لانا`
  - `ابقي` → `لعبداس`

### hf_quran_md_ayah_route_banna_019_009_067745
- bucket: 041_060
- gold: `قالكذلكقالربكهوعليهينوقدخلقتكمنقبلولمتكشيئا`
- pred: `ملانعبدمالارحبدمنالرحيمارحيواالحمنمالرنبدمنوالنبر`
- char_accuracy: 0.093
- lengths gold/pred: 43/49
- chunks: 7
  - `قالكذلك` → `ملانعبد`
  - `قالربك` → `مالارحبدمن`
  - `هوعلي` → `الرحيم`
  - `هينوقد` → `ارحيوا`
  - `خلقتكمن` → `الحمن`
  - `قبلولم` → `مالرنبدمن`
  - `تكشيئا` → `والنبر`

### hf_quran_md_ayah_route_hussary.teacher_036_006_111309
- bucket: 021_040
- gold: `لتنذرقوماماانذراباؤهمفهمغافلون`
- pred: `لالنعباسلرحعبدسدسالراسالخناس`
- char_accuracy: 0.100
- lengths gold/pred: 30/28
- chunks: 4
  - `لتنذرقوما` → `لالنعباس`
  - `ماانذر` → `لرحعبدسدس`
  - `اباؤهمفهم` → `الراس`
  - `غافلون` → `الخناس`

### hf_quran_md_ayah_route_ibrahim_akhdar_051_054_141851
- bucket: 001_020
- gold: `فتولعنهمفماانتبملوم`
- pred: `ملاادتاسعاسالااس`
- char_accuracy: 0.105
- lengths gold/pred: 19/16
- chunks: 3
  - `فتولعنهم` → `ملاادتاس`
  - `فماانت` → `عاس`
  - `بملوم` → `الااس`

### hf_quran_md_ayah_route_abdul_basit_murattal_026_203_094027
- bucket: 001_020
- gold: `فيقولواهلنحنمنظرون`
- pred: `مماععبمد`
- char_accuracy: 0.111
- lengths gold/pred: 18/8
- chunks: 2
  - `فيقولواهل` → `مما`
  - `نحنمنظرون` → `ععبمد`


## Best examples

### hf_quran_md_ayah_route_abdul_basit_murattal_026_053_089527
- gold: `فارسلفرعونفيالمدائنحاشرين`
- pred: `مالنروايانماساكالرحين`
- char_accuracy: 0.440

### hf_quran_md_ayah_route_ghamadi_055_039_148182
- gold: `فيومئذلايسالعنذنبهانسولاجان`
- pred: `لكالناسالرحمناشاناسولااس`
- char_accuracy: 0.407

### hf_quran_md_ayah_route_ghamadi_055_004_147132
- gold: `علمهالبيان`
- pred: `لالناس`
- char_accuracy: 0.400

### hf_quran_md_ayah_route_hussary.teacher_034_036_109239
- gold: `قلانربييبسطالرزقلمنيشاءويقدرولكناكثرالناسلايعلمون`
- pred: `مليالنبدتاالربدملحاسوااسملناسالناسادمن`
- char_accuracy: 0.388

### hf_quran_md_ayah_route_minshawy_mujawwad_007_015_029042
- gold: `قالانكمنالمنظرين`
- pred: `الرعبدسالرنيم`
- char_accuracy: 0.375

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_003_131_012691
- gold: `واتقواالنارالتياعدتللكافرين`
- pred: `لااناالعانالارحمن`
- char_accuracy: 0.370

### hf_quran_md_ayah_route_abdul_basit_murattal_026_071_090067
- gold: `قالوانعبداصنامافنظللهاعاكفين`
- pred: `ملعبدمانانوالنعيمن`
- char_accuracy: 0.357

### hf_quran_md_ayah_route_warsh_yassin_024_009_083976
- gold: `والخامسةانغضباللهعليهاانكانمنالصادقين`
- pred: `ملشادسلخناسالخنيمالرحماارحين`
- char_accuracy: 0.351

### hf_quran_md_ayah_route_alafasy_038_037_120190
- gold: `والشياطينكلبناءوغواص`
- pred: `الالناسولنباس`
- char_accuracy: 0.350

### hf_quran_md_ayah_route_hussary.teacher_036_017_111639
- gold: `وماعليناالاالبلاغالمبين`
- pred: `لكلنعبداالرحاالنح`
- char_accuracy: 0.348

### hf_quran_md_ayah_route_hussary.teacher_037_149_118089
- gold: `فاستفتهمالربكالبناتولهمالبنون`
- pred: `ملكالنعدانالامالخناس`
- char_accuracy: 0.345

### hf_quran_md_ayah_route_abdullah_basfar_008_068_036813
- gold: `لولاكتابمناللهسبقلمسكمفيمااخذتمعذابعظيم`
- pred: `ملانبدمنمالناسالناسعلابدمنابتيم`
- char_accuracy: 0.333

### hf_quran_md_ayah_route_husary_mujawwad_012_095_050704
- gold: `قالواتاللهانكلفيضلالكالقديم`
- pred: `البدتاسمابدمالنيم`
- char_accuracy: 0.333

### hf_quran_md_ayah_route_abdul_basit_murattal_023_061_081997
- gold: `اولئكيسارعونفيالخيراتوهملهاسابقون`
- pred: `ولنامالعبدالوناسارحمن`
- char_accuracy: 0.333

### hf_quran_md_ayah_route_abdul_basit_murattal_023_095_083017
- gold: `واناعلياننريكمانعدهملقادرون`
- pred: `ولنالناسماعبدونالاعبد`
- char_accuracy: 0.333
