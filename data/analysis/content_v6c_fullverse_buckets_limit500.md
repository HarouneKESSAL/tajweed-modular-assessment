# Full-verse content diagnostics

- checkpoint: `checkpoints\content_v6c_old_expanded_full_vocab_hd96.pt`
- manifest: `data\manifests\content_v6a_short_hf_ayah_r1_hf_ayah_clean_all.jsonl`
- split: `val`
- blank_penalty: 0.4

## Overall

- samples: 448
- exact_match: 0.000
- char_accuracy: 0.142
- edit_distance: 23.951
- avg_gold_len: 27.5
- avg_pred_len: 6.0

## Buckets

| bucket | samples | exact | char_accuracy | edit_distance | avg_gold_len | avg_pred_len |
|---|---:|---:|---:|---:|---:|---:|
| 001_020 | 176 | 0.000 | 0.164 | 12.324 | 14.7 | 5.9 |
| 021_040 | 178 | 0.000 | 0.138 | 25.545 | 29.5 | 6.0 |
| 041_060 | 94 | 0.000 | 0.106 | 42.702 | 47.7 | 6.2 |

## Worst examples

### hf_quran_md_ayah_route_saood_ash_shuraym_002_001_000210
- bucket: 001_020
- gold: `الم`
- pred: `لنعيس`
- char_accuracy: 0.000
- lengths gold/pred: 3/5

### hf_quran_md_ayah_route_saood_ash_shuraym_003_001_008790
- bucket: 001_020
- gold: `الم`
- pred: `ولكلحعبدس`
- char_accuracy: 0.000
- lengths gold/pred: 3/9

### hf_quran_md_ayah_route_minshawy_mujawwad_007_139_032762
- bucket: 021_040
- gold: `انهؤلاءمتبرماهمفيهوباطلماكانوايعملون`
- pred: `شدة`
- char_accuracy: 0.000
- lengths gold/pred: 36/3

### hf_quran_md_ayah_route_banna_020_034_071435
- bucket: 001_020
- gold: `ونذكرككثيرا`
- pred: `ملااس`
- char_accuracy: 0.000
- lengths gold/pred: 11/5

### hf_quran_md_ayah_route_abdul_basit_murattal_026_001_087967
- bucket: 001_020
- gold: `طسم`
- pred: `مالحين`
- char_accuracy: 0.000
- lengths gold/pred: 3/6

### hf_quran_md_ayah_route_ali_jaber_028_001_097568
- bucket: 001_020
- gold: `طسم`
- pred: `اياكني`
- char_accuracy: 0.000
- lengths gold/pred: 3/6

### hf_quran_md_ayah_route_ali_jaber_029_001_100208
- bucket: 001_020
- gold: `الم`
- pred: `لناس`
- char_accuracy: 0.000
- lengths gold/pred: 3/4

### hf_quran_md_ayah_route_ali_jaber_030_001_102278
- bucket: 001_020
- gold: `الم`
- pred: `لكالناس`
- char_accuracy: 0.000
- lengths gold/pred: 3/7

### hf_quran_md_ayah_route_ali_jaber_031_001_104078
- bucket: 001_020
- gold: `الم`
- pred: `لشلناس`
- char_accuracy: 0.000
- lengths gold/pred: 3/6

### hf_quran_md_ayah_route_ali_jaber_032_001_105098
- bucket: 001_020
- gold: `الم`
- pred: `لكالناس`
- char_accuracy: 0.000
- lengths gold/pred: 3/7

### hf_quran_md_ayah_route_hussary.teacher_036_001_111159
- bucket: 001_020
- gold: `يس`
- pred: `ملكالناس`
- char_accuracy: 0.000
- lengths gold/pred: 2/8

### hf_quran_md_ayah_route_hussary.teacher_037_179_118989
- bucket: 001_020
- gold: `وابصرفسوفيبصرون`
- pred: `ملكالناس`
- char_accuracy: 0.000
- lengths gold/pred: 15/8

### hf_quran_md_ayah_route_alafasy_042_002_128200
- bucket: 001_020
- gold: `عسق`
- pred: `ملكالنيك`
- char_accuracy: 0.000
- lengths gold/pred: 3/8

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_074_020_165433
- bucket: 001_020
- gold: `ثمقتلكيفقدر`
- pred: `لرحمدماس`
- char_accuracy: 0.000
- lengths gold/pred: 11/8

### hf_quran_md_ayah_route_abu_bakr_ash_shaatree_079_021_171975
- bucket: 001_020
- gold: `فكذبوعصي`
- pred: `الخنان`
- char_accuracy: 0.000
- lengths gold/pred: 8/6

### hf_quran_md_ayah_route_minshawy_mujawwad_008_028_035612
- bucket: 041_060
- gold: `واعلمواانمااموالكمواولادكمفتنةواناللهعندهاجرعظيم`
- pred: `بد`
- char_accuracy: 0.021
- lengths gold/pred: 48/2

### hf_quran_md_ayah_route_minshawy_mujawwad_007_025_029342
- bucket: 021_040
- gold: `قالفيهاتحيونوفيهاتموتونومنهاتخرجون`
- pred: `لعبدس`
- char_accuracy: 0.029
- lengths gold/pred: 34/5

### hf_quran_md_ayah_route_abdul_basit_murattal_027_029_095617
- bucket: 021_040
- gold: `قالتياايهاالملاانيالقياليكتابكريم`
- pred: `لد`
- char_accuracy: 0.030
- lengths gold/pred: 33/2

### hf_quran_md_ayah_route_minshawy_mujawwad_007_109_031862
- bucket: 021_040
- gold: `قالالملامنقومفرعونانهذالساحرعليم`
- pred: `لة`
- char_accuracy: 0.031
- lengths gold/pred: 32/2

### hf_quran_md_ayah_route_saood_ash_shuraym_003_102_011820
- bucket: 041_060
- gold: `ياايهاالذينامنوااتقوااللهحقتقاتهولاتموتنالاوانتممسلمون`
- pred: `لباد`
- char_accuracy: 0.037
- lengths gold/pred: 54/4

### hf_quran_md_ayah_route_minshawy_mujawwad_007_132_032552
- bucket: 041_060
- gold: `وقالوامهماتاتنابهمنايةلتسحرنابهافمانحنلكبمؤمنين`
- pred: `شمن`
- char_accuracy: 0.043
- lengths gold/pred: 47/3

### hf_quran_md_ayah_route_minshawy_mujawwad_006_041_024872
- bucket: 041_060
- gold: `بلاياهتدعونفيكشفماتدعوناليهانشاءوتنسونماتشركون`
- pred: `من`
- char_accuracy: 0.043
- lengths gold/pred: 46/2

### hf_quran_md_ayah_route_warsh_yassin_020_122_074076
- bucket: 021_040
- gold: `ثماجتباهربهفتابعليهوهدي`
- pred: `لناس`
- char_accuracy: 0.043
- lengths gold/pred: 23/4

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_072_012_163753
- bucket: 021_040
- gold: `واناظنناانلننعجزاللهفيالارضولننعجزههربا`
- pred: `لنمس`
- char_accuracy: 0.051
- lengths gold/pred: 39/4

### hf_quran_md_ayah_route_abdullaah_3awwaad_al_juhaynee_070_008_161473
- bucket: 001_020
- gold: `يومتكونالسماءكالمهل`
- pred: `لنحعيدس`
- char_accuracy: 0.053
- lengths gold/pred: 19/7

### hf_quran_md_ayah_route_saood_ash_shuraym_002_278_008520
- bucket: 041_060
- gold: `ياايهاالذينامنوااتقوااللهوذروامابقيمنالرباانكنتممؤمنين`
- pred: `لابدس`
- char_accuracy: 0.056
- lengths gold/pred: 54/5

### hf_quran_md_ayah_route_muhsin_al_qasim_093_005_182506
- bucket: 001_020
- gold: `ولسوفيعطيكربكفترضي`
- pred: `لنحاس`
- char_accuracy: 0.056
- lengths gold/pred: 18/5

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_168_019801
- bucket: 041_060
- gold: `انالذينكفرواوظلموالميكناللهليغفرلهمولاليهديهمطريقا`
- pred: `لواس`
- char_accuracy: 0.060
- lengths gold/pred: 50/4

### hf_quran_md_ayah_route_minshawy_mujawwad_008_008_035012
- bucket: 021_040
- gold: `ليحقالحقويبطلالباطلولوكرهالمجرمون`
- pred: `مابد`
- char_accuracy: 0.061
- lengths gold/pred: 33/4

### hf_quran_md_ayah_route_alafasy_038_036_120160
- bucket: 021_040
- gold: `فسخرنالهالريحتجريبامرهرخاءحيثاصاب`
- pred: `لاندوس`
- char_accuracy: 0.061
- lengths gold/pred: 33/6
