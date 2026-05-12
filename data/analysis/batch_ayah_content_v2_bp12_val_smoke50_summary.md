# Batch ayah content scoring

- manifest: `data\manifests\content_v6a_short_hf_ayah_r1_hf_ayah_clean_all.jsonl`
- split: `val`
- checkpoint: `checkpoints\content_ayah_hf_v2_balanced_hd96.pt`
- decoder_config: `configs\content_ayah_decoder_bp12.json`
- blank_penalty: 1.2

## Overall strict acceptance

- samples: 50
- avg_score: 76.65
- avg_char_accuracy: 0.767
- avg_edit_distance: 9.380
- exact_rate: 0.040
- accepted_rate: 0.040
- acceptance_counts: `{'accepted_exact': 2, 'not_accepted': 48}`
- quality_counts: `{'content_verified_exact': 2, 'likely_same_ayah_but_not_clean': 8, 'same_ayah_candidate_review_required': 30, 'partial_content_match_review_required': 6, 'weak_or_wrong_content': 3, 'almost_correct_review_required': 1}`

## By reciter

| reciter | samples | avg_score | char_acc | edit | exact_rate | accepted_rate |
|---|---:|---:|---:|---:|---:|---:|
| minshawy_mujawwad | 12 | 63.68 | 0.637 | 15.000 | 0.000 | 0.000 |
| abdurrahmaan_as_sudais | 11 | 72.39 | 0.724 | 11.000 | 0.000 | 0.000 |
| husary_mujawwad | 3 | 77.64 | 0.776 | 7.333 | 0.000 | 0.000 |
| abdullah_basfar | 11 | 84.82 | 0.848 | 5.909 | 0.000 | 0.000 |
| saood_ash_shuraym | 13 | 85.09 | 0.851 | 6.231 | 0.154 | 0.154 |

## Worst examples

### hf_quran_md_ayah_route_minshawy_mujawwad_008_028_035612
- reciter: minshawy_mujawwad
- score: 2.08
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `واعلمواانمااموالكمواولادكمفتنةواناللهعندهاجرعظيم`
- pred: `ولنواانمامنتممولاتكفنتنواعدواجينغيوالموامالواكواملاكمفنتوانلعدعروامين`
- edit_distance: 47

### hf_quran_md_ayah_route_minshawy_mujawwad_006_041_024872
- reciter: minshawy_mujawwad
- score: 41.3
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `بلاياهتدعونفيكشفماتدعوناليهانشاءوتنسونماتشركون`
- pred: `بذيتتنتيسكماتدنالهنششكماتدونالهساتانسمماتشركون`
- edit_distance: 27

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_112_018121
- reciter: abdurrahmaan_as_sudais
- score: 46.3
- quality: weak_or_wrong_content
- verdict: not_accepted
- gold: `ومنيكسبخطيئةاواثماثميرمبهبريئافقداحتملبهتاناواثمامبينا`
- pred: `ونايشرصخطياتاعافماثمالرمبيبرياافقدحتملبهتانافقدحتملباتالموافممبيناي`
- edit_distance: 29

### hf_quran_md_ayah_route_minshawy_mujawwad_007_139_032762
- reciter: minshawy_mujawwad
- score: 52.78
- quality: partial_content_match_review_required
- verdict: not_accepted
- gold: `انهؤلاءمتبرماهمفيهوباطلماكانوايعملون`
- pred: `اناءلتمتبرمااننفيدضطنمااكعملون`
- edit_distance: 17

### hf_quran_md_ayah_route_minshawy_mujawwad_008_008_035012
- reciter: minshawy_mujawwad
- score: 63.64
- quality: partial_content_match_review_required
- verdict: not_accepted
- gold: `ليحقالحقويبطلالباطلولوكرهالمجرمون`
- pred: `لحقللحقوايهالقللباطلوالكراامورمون`
- edit_distance: 12

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_068_016801
- reciter: abdurrahmaan_as_sudais
- score: 66.67
- quality: partial_content_match_review_required
- verdict: not_accepted
- gold: `ولهديناهمصراطامستقيما`
- pred: `ولاديناعهمسراقملستقيماا`
- edit_distance: 7

### hf_quran_md_ayah_route_minshawy_mujawwad_007_005_028742
- reciter: minshawy_mujawwad
- score: 67.39
- quality: partial_content_match_review_required
- verdict: not_accepted
- gold: `فماكاندعواهماذجاءهمباسناالاانقالوااناكناظالمين`
- pred: `فمكانتواهماجياءباسناالااانتهاناكناغلنمين`
- edit_distance: 15

### hf_quran_md_ayah_route_minshawy_mujawwad_007_080_030992
- reciter: minshawy_mujawwad
- score: 67.92
- quality: partial_content_match_review_required
- verdict: not_accepted
- gold: `ولوطااذقاللقومهاتاتونالفاحشةماسبقكمبهامناحدمنالعالمين`
- pred: `ولوقااقاللقمذافتوناللثاقسفماسبقكنبذمناحدذمننلعالين`
- edit_distance: 17

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_117_018271
- reciter: abdurrahmaan_as_sudais
- score: 69.77
- quality: partial_content_match_review_required
- verdict: not_accepted
- gold: `انيدعونمندونهالااناثاوانيدعونالاشيطانامريدا`
- pred: `انيدرونننونياائناسوايدمونالاشقانامريدل`
- edit_distance: 13

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_003_131_012691
- reciter: abdurrahmaan_as_sudais
- score: 70.37
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `واتقواالنارالتياعدتللكافرين`
- pred: `وتقنارللتيااتتللكافرين`
- edit_distance: 8

### hf_quran_md_ayah_route_minshawy_mujawwad_007_109_031862
- reciter: minshawy_mujawwad
- score: 71.88
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `قالالملامنقومفرعونانهذالساحرعليم`
- pred: `قاللملرموقومذعونانهذلساحرمنعلين`
- edit_distance: 9

### hf_quran_md_ayah_route_saood_ash_shuraym_003_089_011430
- reciter: saood_ash_shuraym
- score: 72.09
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `الاالذينتابوامنبعدذلكواصلحوافاناللهغفوررحيم`
- pred: `انالذينتابومنبعلدالكوافخلحوفانالاهرفورررحين`
- edit_distance: 12

### hf_quran_md_ayah_route_minshawy_mujawwad_007_025_029342
- reciter: minshawy_mujawwad
- score: 73.53
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `قالفيهاتحيونوفيهاتموتونومنهاتخرجون`
- pred: `قالفياتينوفيااتموتونمناكرلون`
- edit_distance: 9

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_168_019801
- reciter: abdurrahmaan_as_sudais
- score: 74.0
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `انالذينكفرواوظلموالميكناللهليغفرلهمولاليهديهمطريقا`
- pred: `اناللينكبروبلمولميكمنللاهليوغفرلهمولالياهديهمقريقاا`
- edit_distance: 13

### hf_quran_md_ayah_route_saood_ash_shuraym_003_102_011820
- reciter: saood_ash_shuraym
- score: 74.07
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `ياايهاالذينامنوااتقوااللهحقتقاتهولاتموتنالاوانتممسلمون`
- pred: `ياايهلذيناامنااتقالهحقتقتيولاذموتنايالاوانتمللمون`
- edit_distance: 14

### hf_quran_md_ayah_route_saood_ash_shuraym_002_179_005550
- reciter: saood_ash_shuraym
- score: 74.36
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `ولكمفيالقصاصحياةيااوليالالبابلعلكمتتقون`
- pred: `ولكمفلكصاصحاتاياءنلاالبابلعلكمتتقون`
- edit_distance: 10

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_167_019771
- reciter: abdurrahmaan_as_sudais
- score: 74.42
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `انالذينكفرواوصدواعنسبيلاللهقدضلواضلالابعيدا`
- pred: `انالذينكفروصتوعاسبيرالاهقدغلوضلالبعيداي`
- edit_distance: 11

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_003_158_013501
- reciter: abdurrahmaan_as_sudais
- score: 75.0
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `ولئنمتماوقتلتملالياللهتحشرون`
- pred: `ولممتموقتلتملالاللاهتشون`
- edit_distance: 7

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_071_016891
- reciter: abdurrahmaan_as_sudais
- score: 75.51
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `ياايهاالذينامنواخذواحذركمفانفرواثباتاوانفرواجميعا`
- pred: `اياايهلذيناامنوخذوحذركمفانفروذباتاوينفروجنيا`
- edit_distance: 12

### hf_quran_md_ayah_route_saood_ash_shuraym_002_278_008520
- reciter: saood_ash_shuraym
- score: 75.93
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `ياايهاالذينامنوااتقوااللهوذروامابقيمنالرباانكنتممؤمنين`
- pred: `ياايهلذينااماتقلهودرومابقيمناللباانكنتممؤمنين`
- edit_distance: 13

### hf_quran_md_ayah_route_husary_mujawwad_012_074_050074
- reciter: husary_mujawwad
- score: 76.0
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `قالوافماجزاؤهانكنتمكاذبين`
- pred: `قاجوفماجزاااءهانكتمكاذبين`
- edit_distance: 6

### hf_quran_md_ayah_route_minshawy_mujawwad_007_132_032552
- reciter: minshawy_mujawwad
- score: 76.6
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `وقالوامهماتاتنابهمنايةلتسحرنابهافمانحنلكبمؤمنين`
- pred: `وقلمهماتتنابذمنايتلتحرنابافماننلكبممنمين`
- edit_distance: 11

### hf_quran_md_ayah_route_abdullah_basfar_011_009_044433
- reciter: abdullah_basfar
- score: 76.6
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `ولئناذقناالانسانمنارحمةثمنزعناهامنهانهليئوسكفور`
- pred: `وولااذقنالاسانمنارحمتنثمنزعناهامنهانهوليعاوسمكفور`
- edit_distance: 11

### hf_quran_md_ayah_route_husary_mujawwad_012_058_049594
- reciter: husary_mujawwad
- score: 76.92
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `وجاءاخوةيوسففدخلواعليهفعرفهموهملهمنكرون`
- pred: `وجاااخوتويوسففدخلوعليهفرفالهموهملهممكون`
- edit_distance: 9

### hf_quran_md_ayah_route_minshawy_mujawwad_007_204_034712
- reciter: minshawy_mujawwad
- score: 78.05
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `واذاقرئالقرانفاستمعوالهوانصتوالعلكمترحمون`
- pred: `اذاقرااقروانفاستمعولواصتولعلكمتررحمون`
- edit_distance: 9

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_005_030_020941
- reciter: abdurrahmaan_as_sudais
- score: 78.95
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `فطوعتلهنفسهقتلاخيهفقتلهفاصبحمنالخاسرين`
- pred: `فقواعتلغونفساقةلاخيهفقتلهوفاصبحمنالصاسرين`
- edit_distance: 8

### hf_quran_md_ayah_route_saood_ash_shuraym_003_082_011220
- reciter: saood_ash_shuraym
- score: 79.31
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `فمنتوليبعدذلكفاولئكهمالفاسقون`
- pred: `فمنتوالابعدذالكفعلائكهمالفاسقون`
- edit_distance: 6

### hf_quran_md_ayah_route_abdullah_basfar_008_068_036813
- reciter: abdullah_basfar
- score: 79.49
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `لولاكتابمناللهسبقلمسكمفيمااخذتمعذابعظيم`
- pred: `لولاككابمناليسبقلمالسكمفيمااخلتمعذابعظييمن`
- edit_distance: 8

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_006_009_023911
- reciter: abdurrahmaan_as_sudais
- score: 79.55
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `ولوجعلناهملكالجعلناهرجلاوللبسناعليهممايلبسون`
- pred: `ولوجعلناهملكلنجعالماهرجنوللبسناعليمماايابسون`
- edit_distance: 9

### hf_quran_md_ayah_route_husary_mujawwad_012_057_049564
- reciter: husary_mujawwad
- score: 80.0
- quality: same_ayah_candidate_review_required
- verdict: not_accepted
- gold: `ولاجرالاخرةخيرللذينامنواوكانوايتقون`
- pred: `ولاجرالااخرةخييرلهلذيناامنوكانويتقون`
- edit_distance: 7
