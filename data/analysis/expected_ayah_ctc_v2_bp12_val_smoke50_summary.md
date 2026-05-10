# Expected ayah CTC scoring

- manifest: `data\manifests\content_v6a_short_hf_ayah_r1_hf_ayah_clean_all.jsonl`
- split: `val`
- checkpoint: `checkpoints\content_ayah_hf_v2_balanced_hd96.pt`
- decoder_config: `configs\content_ayah_decoder_bp12.json`
- blank_penalty: 1.2

## Overall

- samples: `50`
- free_exact_rate: `0.04`
- expected_text_accepted_rate: `0.06`
- expected_text_strong_review_rate: `0.34`
- expected_text_plausible_review_rate: `0.38`
- avg_free_char_accuracy: `0.7665170418571995`
- avg_expected_ctc_loss_per_char: `0.8071786148420026`
- avg_expected_ctc_confidence: `0.4884756937222736`
- verdict_counts: `{'accepted_free_decode_exact': 2, 'expected_text_strong_but_review_required': 17, 'expected_text_plausible_review_required': 19, 'free_decode_similarity_review_required': 4, 'not_supported': 7, 'accepted_expected_text_near_exact_review_recommended': 1}`

## By reciter

| reciter | samples | free exact | exp accepted | exp strong review | free char | exp loss/char | exp confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| abdullah_basfar | 11 | 0.000 | 0.091 | 0.636 | 0.848 | 0.530 | 0.598 |
| saood_ash_shuraym | 13 | 0.154 | 0.154 | 0.462 | 0.851 | 0.560 | 0.592 |
| husary_mujawwad | 3 | 0.000 | 0.000 | 0.667 | 0.776 | 0.611 | 0.548 |
| abdurrahmaan_as_sudais | 11 | 0.000 | 0.000 | 0.091 | 0.724 | 0.983 | 0.398 |
| minshawy_mujawwad | 12 | 0.000 | 0.000 | 0.083 | 0.637 | 1.218 | 0.345 |

## Best expected-text support

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

### hf_quran_md_ayah_route_abdullah_basfar_011_079_046533
- reciter: abdullah_basfar
- verdict: accepted_expected_text_near_exact_review_recommended
- free_score: 97.67
- free_exact: False
- expected_loss_per_char: 0.2309991481692292
- expected_confidence: 0.7937401421672319
- gold: `قالوالقدعلمتمالنافيبناتكمنحقوانكلتعلممانريد`
- free_prediction: `قالولقدعلمتمالنافيبناتكمنحقوانكلتعلممانريد`

### hf_quran_md_ayah_route_saood_ash_shuraym_002_077_002490
- reciter: saood_ash_shuraym
- verdict: expected_text_strong_but_review_required
- free_score: 94.44
- free_exact: False
- expected_loss_per_char: 0.26762252383761936
- expected_confidence: 0.7651965700586941
- gold: `اولايعلموناناللهيعلممايسرونومايعلنون`
- free_prediction: `اولايعلموناناللاهيعلممايسرونومايوعلنون`

### hf_quran_md_ayah_route_abdullah_basfar_010_086_043473
- reciter: abdullah_basfar
- verdict: expected_text_strong_but_review_required
- free_score: 92.31
- free_exact: False
- expected_loss_per_char: 0.27602335122915417
- expected_confidence: 0.7587952117534912
- gold: `ونجنابرحمتكمنالقومالكافرين`
- free_prediction: `ونجنابرحمتكمنالقوميكافرين`

### hf_quran_md_ayah_route_minshawy_mujawwad_007_015_029042
- reciter: minshawy_mujawwad
- verdict: expected_text_strong_but_review_required
- free_score: 87.5
- free_exact: False
- expected_loss_per_char: 0.40811049938201904
- expected_confidence: 0.6649054031268206
- gold: `قالانكمنالمنظرين`
- free_prediction: `قالانكامنالنظرين`

### hf_quran_md_ayah_route_abdullah_basfar_009_057_038733
- reciter: abdullah_basfar
- verdict: expected_text_strong_but_review_required
- free_score: 84.09
- free_exact: False
- expected_loss_per_char: 0.4099030928178267
- expected_confidence: 0.6637145657282797
- gold: `لويجدونملجااومغاراتاومدخلالولوااليهوهميجمحون`
- free_prediction: `الويجدونمالجااومغاراتةماومودخلالوالواليهوهميجمحون`

### hf_quran_md_ayah_route_saood_ash_shuraym_002_018_000720
- reciter: saood_ash_shuraym
- verdict: expected_text_strong_but_review_required
- free_score: 84.21
- free_exact: False
- expected_loss_per_char: 0.41511510547838715
- expected_confidence: 0.66026427629542
- gold: `صمبكمعميفهملايرجعون`
- free_prediction: `ثممبكمعمينفهملايرجعون`

### hf_quran_md_ayah_route_abdullah_basfar_009_078_039363
- reciter: abdullah_basfar
- verdict: expected_text_strong_but_review_required
- free_score: 82.98
- free_exact: False
- expected_loss_per_char: 0.423636091516373
- expected_confidence: 0.6546620756491061
- gold: `الميعلموااناللهيعلمسرهمونجواهمواناللهعلامالغيوب`
- free_prediction: `اليعلمماناللهيعلمسرهونجواهمواناللاهعلاملخيوبن`

### hf_quran_md_ayah_route_saood_ash_shuraym_002_004_000300
- reciter: saood_ash_shuraym
- verdict: expected_text_strong_but_review_required
- free_score: 88.46
- free_exact: False
- expected_loss_per_char: 0.46143234693087065
- expected_confidence: 0.6303800755820613
- gold: `والذينيؤمنونبماانزلاليكوماانزلمنقبلكوبالاخرةهميوقنون`
- free_prediction: `ولذينيؤمنونبماهنزلاليكوماانزلمنقبلنكوبالااخلتهميوقنون`

### hf_quran_md_ayah_route_husary_mujawwad_012_057_049564
- reciter: husary_mujawwad
- verdict: expected_text_strong_but_review_required
- free_score: 80.0
- free_exact: False
- expected_loss_per_char: 0.4766501835414341
- expected_confidence: 0.6208596782439941
- gold: `ولاجرالاخرةخيرللذينامنواوكانوايتقون`
- free_prediction: `ولاجرالااخرةخييرلهلذيناامنوكانويتقون`

### hf_quran_md_ayah_route_abdullah_basfar_008_055_036423
- reciter: abdullah_basfar
- verdict: expected_text_strong_but_review_required
- free_score: 84.21
- free_exact: False
- expected_loss_per_char: 0.49736449592991877
- expected_confidence: 0.6081312820317758
- gold: `انشرالدوابعنداللهالذينكفروافهملايؤمنون`
- free_prediction: `انشردوابعنداللهلذينكفرافهملايمنلون`

### hf_quran_md_ayah_route_saood_ash_shuraym_002_056_001860
- reciter: saood_ash_shuraym
- verdict: expected_text_strong_but_review_required
- free_score: 93.33
- free_exact: False
- expected_loss_per_char: 0.5229461352030437
- expected_confidence: 0.5927715876566999
- gold: `ثمبعثناكممنبعدموتكملعلكمتشكرون`
- free_prediction: `ثمبعثناكمنبعدمعتكملعلكمتشكرون`

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_138_018901
- reciter: abdurrahmaan_as_sudais
- verdict: expected_text_strong_but_review_required
- free_score: 85.71
- free_exact: False
- expected_loss_per_char: 0.5362162590026855
- expected_confidence: 0.5849573976086695
- gold: `بشرالمنافقينبانلهمعذابااليما`
- free_prediction: `ابشرلمنافقينباللهمعذاباالينا`

### hf_quran_md_ayah_route_saood_ash_shuraym_002_066_002160
- reciter: saood_ash_shuraym
- verdict: expected_text_strong_but_review_required
- free_score: 86.67
- free_exact: False
- expected_loss_per_char: 0.5479111565483941
- expected_confidence: 0.5781562277836915
- gold: `فجعلناهانكالالمابينيديهاوماخلفهاوموعظةللمتقين`
- free_prediction: `فجعلناهانكالالمابينيديهوماخلفهاومويغتالالمتقين`

### hf_quran_md_ayah_route_abdullah_basfar_011_077_046473
- reciter: abdullah_basfar
- verdict: expected_text_strong_but_review_required
- free_score: 81.25
- free_exact: False
- expected_loss_per_char: 0.550846258799235
- expected_confidence: 0.5764617680609151
- gold: `ولماجاءترسلنالوطاسيءبهموضاقبهمذرعاوقالهذايومعصيب`
- free_prediction: `ولماجاترسلنالوقنسيابهوضاقبهمذرععوقاولهاذايومنعصيب`

### hf_quran_md_ayah_route_husary_mujawwad_012_074_050074
- reciter: husary_mujawwad
- verdict: expected_text_strong_but_review_required
- free_score: 76.0
- free_exact: False
- expected_loss_per_char: 0.5590812301635742
- expected_confidence: 0.5717341146704663
- gold: `قالوافماجزاؤهانكنتمكاذبين`
- free_prediction: `قاجوفماجزاااءهانكتمكاذبين`

### hf_quran_md_ayah_route_abdullah_basfar_011_115_047613
- reciter: abdullah_basfar
- verdict: expected_text_strong_but_review_required
- free_score: 89.66
- free_exact: False
- expected_loss_per_char: 0.608908488832671
- expected_confidence: 0.5439442664068657
- gold: `واصبرفاناللهلايضيعاجرالمحسنين`
- free_prediction: `واصبرفاناللالايضبيعاجرعلمحسنين`

### hf_quran_md_ayah_route_saood_ash_shuraym_002_042_001440
- reciter: saood_ash_shuraym
- verdict: expected_text_strong_but_review_required
- free_score: 83.33
- free_exact: False
- expected_loss_per_char: 0.6320149557931083
- expected_confidence: 0.5315197325201211
- gold: `ولاتلبسواالحقبالباطلوتكتمواالحقوانتمتعلمون`
- free_prediction: `ولاتمبسالحقبالباطليوتاكتمالحقوانتمتعلمون`

### hf_quran_md_ayah_route_abdullah_basfar_011_022_044823
- reciter: abdullah_basfar
- verdict: expected_text_strong_but_review_required
- free_score: 81.48
- free_exact: False
- expected_loss_per_char: 0.6485106856734665
- expected_confidence: 0.5228238462666666
- gold: `لاجرمانهمفيالاخرةهمالاخسرون`
- free_prediction: `لاجربانهمفياللافرتهوالاخسرون`


## Worst expected-text support

### hf_quran_md_ayah_route_minshawy_mujawwad_008_028_035612
- reciter: minshawy_mujawwad
- verdict: not_supported
- free_score: 2.08
- expected_loss_per_char: 2.9865566889444985
- expected_confidence: 0.050460890466632644
- gold: `واعلمواانمااموالكمواولادكمفتنةواناللهعندهاجرعظيم`
- free_prediction: `ولنواانمامنتممولاتكفنتنواعدواجينغيوالموامالواكواملاكمفنتوانلعدعروامين`

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_112_018121
- reciter: abdurrahmaan_as_sudais
- verdict: not_supported
- free_score: 46.3
- expected_loss_per_char: 2.093628353542752
- expected_confidence: 0.12323916834365968
- gold: `ومنيكسبخطيئةاواثماثميرمبهبريئافقداحتملبهتاناواثمامبينا`
- free_prediction: `ونايشرصخطياتاعافماثمالرمبيبرياافقدحتملبهتانافقدحتملباتالموافممبيناي`

### hf_quran_md_ayah_route_minshawy_mujawwad_006_041_024872
- reciter: minshawy_mujawwad
- verdict: not_supported
- free_score: 41.3
- expected_loss_per_char: 1.9951516856317935
- expected_confidence: 0.1359930244144424
- gold: `بلاياهتدعونفيكشفماتدعوناليهانشاءوتنسونماتشركون`
- free_prediction: `بذيتتنتيسكماتدنالهنششكماتدونالهساتانسمماتشركون`

### hf_quran_md_ayah_route_minshawy_mujawwad_007_139_032762
- reciter: minshawy_mujawwad
- verdict: not_supported
- free_score: 52.78
- expected_loss_per_char: 1.3731635411580403
- expected_confidence: 0.25330435193819595
- gold: `انهؤلاءمتبرماهمفيهوباطلماكانوايعملون`
- free_prediction: `اناءلتمتبرمااننفيدضطنمااكعملون`

### hf_quran_md_ayah_route_minshawy_mujawwad_007_080_030992
- reciter: minshawy_mujawwad
- verdict: not_supported
- free_score: 67.92
- expected_loss_per_char: 1.239516924012382
- expected_confidence: 0.2895240462770696
- gold: `ولوطااذقاللقومهاتاتونالفاحشةماسبقكمبهامناحدمنالعالمين`
- free_prediction: `ولوقااقاللقمذافتوناللثاقسفماسبقكنبذمناحدذمننلعالين`

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_003_158_013501
- reciter: abdurrahmaan_as_sudais
- verdict: free_decode_similarity_review_required
- free_score: 75.0
- expected_loss_per_char: 1.200068746294294
- expected_confidence: 0.3011735066379839
- gold: `ولئنمتماوقتلتملالياللهتحشرون`
- free_prediction: `ولممتموقتلتملالاللاهتشون`

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_003_131_012691
- reciter: abdurrahmaan_as_sudais
- verdict: free_decode_similarity_review_required
- free_score: 70.37
- expected_loss_per_char: 1.087845625700774
- expected_confidence: 0.33694161068598416
- gold: `واتقواالنارالتياعدتللكافرين`
- free_prediction: `وتقنارللتيااتتللكافرين`

### hf_quran_md_ayah_route_minshawy_mujawwad_008_008_035012
- reciter: minshawy_mujawwad
- verdict: not_supported
- free_score: 63.64
- expected_loss_per_char: 1.0832971515077534
- expected_confidence: 0.3384776716252676
- gold: `ليحقالحقويبطلالباطلولوكرهالمجرمون`
- free_prediction: `لحقللحقوايهالقللباطلوالكراامورمون`

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_117_018271
- reciter: abdurrahmaan_as_sudais
- verdict: not_supported
- free_score: 69.77
- expected_loss_per_char: 1.0640225964923238
- expected_confidence: 0.34506495769761997
- gold: `انيدعونمندونهالااناثاوانيدعونالاشيطانامريدا`
- free_prediction: `انيدرونننونياائناسوايدمونالاشقانامريدل`

### hf_quran_md_ayah_route_minshawy_mujawwad_007_109_031862
- reciter: minshawy_mujawwad
- verdict: free_decode_similarity_review_required
- free_score: 71.88
- expected_loss_per_char: 1.0449293851852417
- expected_confidence: 0.3517166549076121
- gold: `قالالملامنقومفرعونانهذالساحرعليم`
- free_prediction: `قاللملرموقومذعونانهذلساحرمنعلين`

### hf_quran_md_ayah_route_minshawy_mujawwad_007_025_029342
- reciter: minshawy_mujawwad
- verdict: free_decode_similarity_review_required
- free_score: 73.53
- expected_loss_per_char: 1.0335425208596623
- expected_confidence: 0.3557444934426242
- gold: `قالفيهاتحيونوفيهاتموتونومنهاتخرجون`
- free_prediction: `قالفياتينوفيااتموتونمناكرلون`

### hf_quran_md_ayah_route_minshawy_mujawwad_007_005_028742
- reciter: minshawy_mujawwad
- verdict: expected_text_plausible_review_required
- free_score: 67.39
- expected_loss_per_char: 0.9433659263279127
- expected_confidence: 0.389315221163678
- gold: `فماكاندعواهماذجاءهمباسناالاانقالوااناكناظالمين`
- free_prediction: `فمكانتواهماجياءباسناالااانتهاناكناغلنمين`

### hf_quran_md_ayah_route_saood_ash_shuraym_003_102_011820
- reciter: saood_ash_shuraym
- verdict: expected_text_plausible_review_required
- free_score: 74.07
- expected_loss_per_char: 0.9417941481978805
- expected_confidence: 0.38992761946504334
- gold: `ياايهاالذينامنوااتقوااللهحقتقاتهولاتموتنالاوانتممسلمون`
- free_prediction: `ياايهلذيناامنااتقالهحقتقتيولاذموتنايالاوانتمللمون`

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_167_019771
- reciter: abdurrahmaan_as_sudais
- verdict: expected_text_plausible_review_required
- free_score: 74.42
- expected_loss_per_char: 0.9062297732331032
- expected_confidence: 0.40404469609857474
- gold: `انالذينكفرواوصدواعنسبيلاللهقدضلواضلالابعيدا`
- free_prediction: `انالذينكفروصتوعاسبيرالاهقدغلوضلالبعيداي`

### hf_quran_md_ayah_route_saood_ash_shuraym_003_089_011430
- reciter: saood_ash_shuraym
- verdict: expected_text_plausible_review_required
- free_score: 72.09
- expected_loss_per_char: 0.8809913812681686
- expected_confidence: 0.41437190743720087
- gold: `الاالذينتابوامنبعدذلكواصلحوافاناللهغفوررحيم`
- free_prediction: `انالذينتابومنبعلدالكوافخلحوفانالاهرفورررحين`

### hf_quran_md_ayah_route_minshawy_mujawwad_007_204_034712
- reciter: minshawy_mujawwad
- verdict: expected_text_plausible_review_required
- free_score: 78.05
- expected_loss_per_char: 0.8800657318859566
- expected_confidence: 0.4147556481145866
- gold: `واذاقرئالقرانفاستمعوالهوانصتوالعلكمترحمون`
- free_prediction: `اذاقرااقروانفاستمعولواصتولعلكمتررحمون`

### hf_quran_md_ayah_route_minshawy_mujawwad_007_132_032552
- reciter: minshawy_mujawwad
- verdict: expected_text_plausible_review_required
- free_score: 76.6
- expected_loss_per_char: 0.876984454215841
- expected_confidence: 0.41603559635696424
- gold: `وقالوامهماتاتنابهمنايةلتسحرنابهافمانحنلكبمؤمنين`
- free_prediction: `وقلمهماتتنابذمنايتلتحرنابافماننلكبممنمين`

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_071_016891
- reciter: abdurrahmaan_as_sudais
- verdict: expected_text_plausible_review_required
- free_score: 75.51
- expected_loss_per_char: 0.8652972785794005
- expected_confidence: 0.4209264016231798
- gold: `ياايهاالذينامنواخذواحذركمفانفرواثباتاوانفرواجميعا`
- free_prediction: `اياايهلذيناامنوخذوحذركمفانفروذباتاوينفروجنيا`

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_004_068_016801
- reciter: abdurrahmaan_as_sudais
- verdict: expected_text_plausible_review_required
- free_score: 66.67
- expected_loss_per_char: 0.8650343758719308
- expected_confidence: 0.4210370788618474
- gold: `ولهديناهمصراطامستقيما`
- free_prediction: `ولاديناعهمسراقملستقيماا`

### hf_quran_md_ayah_route_saood_ash_shuraym_002_179_005550
- reciter: saood_ash_shuraym
- verdict: expected_text_plausible_review_required
- free_score: 74.36
- expected_loss_per_char: 0.8277412805801783
- expected_confidence: 0.4370353124715461
- gold: `ولكمفيالقصاصحياةيااوليالالبابلعلكمتتقون`
- free_prediction: `ولكمفلكصاصحاتاياءنلاالبابلعلكمتتقون`
