# Full-verse as chunks diagnostics

- checkpoint: `checkpoints\content_v6c_old_expanded_full_vocab_hd96.pt`
- manifest: `data\manifests\content_v6a_short_hf_ayah_r1_hf_ayah_clean_all.jsonl`
- split: `val`
- limit: 300
- max_words_per_chunk: 3
- min_window_sec: 0.35
- blank_penalty: 0.4
- window_rows: 844

## Overall

- samples: 300
- exact_match: 0.000
- char_accuracy: 0.214
- edit_distance: 24.283
- avg_gold_len: 31.290
- avg_pred_len: 16.370
- avg_chunks: 2.813

## Buckets

| bucket | samples | exact | char_accuracy | edit_distance | avg_gold_len | avg_pred_len | avg_chunks |
|---|---:|---:|---:|---:|---:|---:|---:|
| 001_020 | 72 | 0.000 | 0.165 | 12.556 | 14.9 | 9.3 | 1.6 |
| 021_040 | 143 | 0.000 | 0.222 | 23.147 | 29.8 | 15.4 | 2.6 |
| 041_060 | 85 | 0.000 | 0.241 | 36.129 | 47.6 | 24.0 | 4.2 |

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
- pred: `ملكالناس`
- char_accuracy: 0.000
- lengths gold/pred: 15/8
- chunks: 1
  - `وابصرفسوفيبصرون` → `ملكالناس`

### hf_quran_md_ayah_route_alafasy_042_002_128200
- bucket: 001_020
- gold: `عسق`
- pred: `ملكالنيك`
- char_accuracy: 0.000
- lengths gold/pred: 3/8
- chunks: 1
  - `عسق` → `ملكالنيك`

### hf_quran_md_ayah_route_ibrahim_akhdar_053_003_143591
- bucket: 001_020
- gold: `وماينطقعنالهوي`
- pred: `لبدمانبتس`
- char_accuracy: 0.071
- lengths gold/pred: 14/9
- chunks: 2
  - `وماينطقعن` → `لبد`
  - `الهوي` → `مانبتس`

### hf_quran_md_ayah_route_hussary.teacher_037_049_115089
- bucket: 001_020
- gold: `كانهنبيضمكنون`
- pred: `ملاناس`
- char_accuracy: 0.077
- lengths gold/pred: 13/6
- chunks: 1
  - `كانهنبيضمكنون` → `ملاناس`

### hf_quran_md_ayah_route_ghamadi_053_051_145032
- bucket: 001_020
- gold: `وثمودفماابقي`
- pred: `لنا`
- char_accuracy: 0.083
- lengths gold/pred: 12/3
- chunks: 1
  - `وثمودفماابقي` → `لنا`

### hf_quran_md_ayah_route_warsh_yassin_020_122_074076
- bucket: 021_040
- gold: `ثماجتباهربهفتابعليهوهدي`
- pred: `لالمنولاناس`
- char_accuracy: 0.087
- lengths gold/pred: 23/11
- chunks: 2
  - `ثماجتباهربه` → `لالمن`
  - `فتابعليهوهدي` → `ولاناس`

### hf_quran_md_ayah_route_abdul_basit_murattal_026_207_094147
- bucket: 021_040
- gold: `مااغنيعنهمماكانوايمتعون`
- pred: `لبداعد`
- char_accuracy: 0.087
- lengths gold/pred: 23/6
- chunks: 2
  - `مااغنيعنهم` → `لبدا`
  - `ماكانوايمتعون` → `عد`

### hf_quran_md_ayah_route_ibrahim_akhdar_053_006_143681
- bucket: 001_020
- gold: `ذومرةفاستوي`
- pred: `لالدتس`
- char_accuracy: 0.091
- lengths gold/pred: 11/6
- chunks: 1
  - `ذومرةفاستوي` → `لالدتس`

### hf_quran_md_ayah_route_warsh_yassin_020_107_073626
- bucket: 001_020
- gold: `لاتريفيهاعوجاولاامتا`
- pred: `ولخناسلخناس`
- char_accuracy: 0.100
- lengths gold/pred: 20/11
- chunks: 2
  - `لاتريفيها` → `ولخناس`
  - `عوجاولاامتا` → `لخناس`

### hf_quran_md_ayah_route_ghamadi_056_017_149862
- bucket: 001_020
- gold: `يطوفعليهمولدانمخلدون`
- pred: `لانبدسالرحاس`
- char_accuracy: 0.100
- lengths gold/pred: 20/12
- chunks: 2
  - `يطوفعليهمولدان` → `لانبدس`
  - `مخلدون` → `الرحاس`

### hf_quran_md_ayah_route_abdul_basit_murattal_026_076_090217
- bucket: 001_020
- gold: `انتمواباؤكمالاقدمون`
- pred: `لا`
- char_accuracy: 0.105
- lengths gold/pred: 19/2
- chunks: 1
  - `انتمواباؤكمالاقدمون` → `لا`

### hf_quran_md_ayah_route_abdul_basit_murattal_026_170_093037
- bucket: 001_020
- gold: `فنجيناهواهلهاجمعين`
- pred: `ولحبداد`
- char_accuracy: 0.111
- lengths gold/pred: 18/7
- chunks: 1
  - `فنجيناهواهلهاجمعين` → `ولحبداد`

### hf_quran_md_ayah_route_abdul_basit_murattal_026_203_094027
- bucket: 001_020
- gold: `فيقولواهلنحنمنظرون`
- pred: `معبدونماعبدمد`
- char_accuracy: 0.111
- lengths gold/pred: 18/13
- chunks: 2
  - `فيقولواهلنحن` → `معبدون`
  - `منظرون` → `ماعبدمد`

### hf_quran_md_ayah_route_ghamadi_053_019_144072
- bucket: 001_020
- gold: `افرايتماللاتوالعزي`
- pred: `لناس`
- char_accuracy: 0.111
- lengths gold/pred: 18/4
- chunks: 1
  - `افرايتماللاتوالعزي` → `لناس`

### hf_quran_md_ayah_route_minshawy_mujawwad_007_025_029342
- bucket: 021_040
- gold: `قالفيهاتحيونوفيهاتموتونومنهاتخرجون`
- pred: `لرعبدماعبدمملناس`
- char_accuracy: 0.118
- lengths gold/pred: 34/16
- chunks: 3
  - `قالفيهاتحيون` → `لرعبد`
  - `وفيهاتموتونومنها` → `ماعبدم`
  - `تخرجون` → `ملناس`


## Best examples

### hf_quran_md_ayah_route_ghamadi_055_004_147132
- gold: `علمهالبيان`
- pred: `لالناس`
- char_accuracy: 0.400

### hf_quran_md_ayah_route_alafasy_038_054_120700
- gold: `انهذالرزقنامالهمننفاد`
- pred: `لالناسالناسالخنان`
- char_accuracy: 0.381

### hf_quran_md_ayah_route_ghamadi_055_039_148182
- gold: `فيومئذلايسالعنذنبهانسولاجان`
- pred: `ملكالناسوالناسولااس`
- char_accuracy: 0.370

### hf_quran_md_ayah_route_hussary.teacher_037_130_117519
- gold: `سلامعليالياسين`
- pred: `ملالناسالم`
- char_accuracy: 0.357

### hf_quran_md_ayah_route_abdul_basit_murattal_026_087_090547
- gold: `ولاتخزنييوميبعثون`
- pred: `مانادمابون`
- char_accuracy: 0.353

### hf_quran_md_ayah_route_hussary.teacher_034_036_109239
- gold: `قلانربييبسطالرزقلمنيشاءويقدرولكناكثرالناسلايعلمون`
- pred: `اليانعبادوالنبدمواعبدالناسادمن`
- char_accuracy: 0.347

### hf_quran_md_ayah_route_abdul_basit_murattal_025_067_087637
- gold: `والذيناذاانفقوالميسرفواولميقترواوكانبينذلكقواما`
- pred: `والناسولحاسوالباسالاا`
- char_accuracy: 0.340

### hf_quran_md_ayah_route_abdullah_basfar_010_008_041133
- gold: `اولئكماواهمالناربماكانوايكسبون`
- pred: `ملانماواكالناس`
- char_accuracy: 0.333

### hf_quran_md_ayah_route_alafasy_043_009_130000
- gold: `ولئنسالتهممنخلقالسماواتوالارضليقولنخلقهنالعزيزالعليم`
- pred: `لنبدتاسمنالناسمالنادالرحيم`
- char_accuracy: 0.327

### hf_quran_md_ayah_route_warsh_yassin_024_005_083856
- gold: `الاالذينتابوامنبعدذلكواصلحوافاناللهغفوررحيم`
- pred: `لامامانالخناسالرحيم`
- char_accuracy: 0.326

### hf_quran_md_ayah_route_abdul_basit_murattal_026_053_089527
- gold: `فارسلفرعونفيالمدائنحاشرين`
- pred: `مالةوالنين`
- char_accuracy: 0.320

### hf_quran_md_ayah_route_alafasy_043_027_130540
- gold: `الاالذيفطرنيفانهسيهدين`
- pred: `الكالنمنالرحي`
- char_accuracy: 0.318

### hf_quran_md_ayah_route_abdul_basit_murattal_026_088_090577
- gold: `يوملاينفعمالولابنون`
- pred: `ملرنباماا`
- char_accuracy: 0.316

### hf_quran_md_ayah_route_banna_016_128_060845
- gold: `اناللهمعالذيناتقواوالذينهممحسنون`
- pred: `ملالبادمالنينمالانعبوس`
- char_accuracy: 0.312

### hf_quran_md_ayah_route_warsh_yassin_023_034_081186
- gold: `ولئناطعتمبشرامثلكمانكماذالخاسرون`
- pred: `ملاماسعابدتماملاود`
- char_accuracy: 0.312
