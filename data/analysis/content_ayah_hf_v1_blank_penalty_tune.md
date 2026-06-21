# Content decoder tuning

- checkpoint: `checkpoints\content_ayah_hf_v1_hd96.pt`
- manifest: `data\manifests\content_ayah_hf_le60.jsonl`
- split: `val`
- split_mode: `text`
- samples: 608

## Results

| blank_penalty | exact | char_accuracy | edit_distance | avg_pred_len |
|---:|---:|---:|---:|---:|
| 0.00 | 0.020 | 0.727 | 7.291 | 22.86 |
| 0.20 | 0.023 | 0.735 | 7.046 | 23.42 |
| 0.40 | 0.023 | 0.741 | 6.873 | 23.97 |
| 0.60 | 0.020 | 0.744 | 6.775 | 24.50 |
| 0.80 | 0.021 | 0.749 | 6.661 | 25.12 |
| 1.00 | 0.021 | 0.752 | 6.567 | 25.68 |
| 1.20 | 0.023 | 0.753 | 6.533 | 26.30 |
| 1.60 | 0.018 | 0.746 | 6.715 | 27.59 |
| 2.00 | 0.018 | 0.731 | 7.081 | 28.87 |
| 2.40 | 0.015 | 0.706 | 7.750 | 30.23 |
| 2.80 | 0.010 | 0.678 | 8.719 | 31.69 |
| 3.20 | 0.007 | 0.651 | 9.933 | 33.26 |

## Best

- blank_penalty: 1.2
- exact_match: 0.023
- char_accuracy: 0.753
- edit_distance: 6.533

## Example errors

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_003_131_012691
- gold: `واتقواالنارالتياعدتللكافرين`
- pred: `ووتقنارللتيااتترلكيافرين`
- char_accuracy: 0.630

### hf_quran_md_ayah_route_abdurrahmaan_as_sudais_005_098_022981
- gold: `اعلموااناللهشديدالعقابواناللهغفوررحيم`
- pred: `للمانمنللااحبيهاللقابوانممهرررررالن`
- char_accuracy: 0.432

### hf_quran_md_ayah_route_minshawy_mujawwad_006_041_024872
- gold: `بلاياهتدعونفيكشفماتدعوناليهانشاءوتنسونماتشركون`
- pred: `بذيتترنفيسكماادانالهيشيشكماتدنالهساافااسممااةشركون`
- char_accuracy: 0.283

### hf_quran_md_ayah_route_minshawy_mujawwad_006_049_025112
- gold: `والذينكذبواباياتنايمسهمالعذاببماكانوايفسقون`
- pred: `والذينكلهبياياكنيمسهالعذاابمكانيسخون`
- char_accuracy: 0.674

### hf_quran_md_ayah_route_minshawy_mujawwad_006_082_026102
- gold: `الذينامنواولميلبسواايمانهمبظلماولئكلهمالامنوهممهتدون`
- pred: `الذيناامنولميلبلسوايمالهمبهلمناولاركلهممانوهنتدون`
- char_accuracy: 0.692

### hf_quran_md_ayah_route_minshawy_mujawwad_006_129_027512
- gold: `وكذلكنوليبعضالظالمينبعضابماكانوايكسبون`
- pred: `وكذنكولليبعببينمينباضبماكاايسفون`
- char_accuracy: 0.605

### hf_quran_md_ayah_route_minshawy_mujawwad_007_002_028652
- gold: `كتابانزلاليكفلايكنفيصدركحرجمنهلتنذربهوذكريللمؤمنين`
- pred: `بهنذلجلكفلاكمفيصاركحرجمننولكنذربهوذقرالومنين`
- char_accuracy: 0.640

### hf_quran_md_ayah_route_minshawy_mujawwad_007_021_029222
- gold: `وقاسمهماانيلكمالمنالناصحين`
- pred: `ققسممنيلكامالاننصيحين`
- char_accuracy: 0.577

### hf_quran_md_ayah_route_minshawy_mujawwad_007_023_029282
- gold: `قالاربناظلمناانفسناوانلمتغفرلناوترحمنالنكوننمنالخاسرين`
- pred: `قلاوبلااهلمناشذاووااناتصفنانتلحناولتفيرننتكلللنكلننناالقان`
- char_accuracy: 0.352

### hf_quran_md_ayah_route_minshawy_mujawwad_007_083_031082
- gold: `فانجيناهواهلهالاامراتهكانتمنالغابرين`
- pred: `فانيناهههلالاموراتهكاننناابرينر`
- char_accuracy: 0.694

### hf_quran_md_ayah_route_minshawy_mujawwad_007_107_031802
- gold: `فالقيعصاهفاذاهيثعبانمبين`
- pred: `فااقاصهفجمذههيثعبانمدين`
- char_accuracy: 0.667

### hf_quran_md_ayah_route_minshawy_mujawwad_007_130_032492
- gold: `ولقداخذناالفرعونبالسنينونقصمنالثمراتلعلهميذكرون`
- pred: `ولقداخرنعالقللععانبسينوناضسمنفماتيلععلهيلكن`
- char_accuracy: 0.511

### hf_quran_md_ayah_route_minshawy_mujawwad_007_135_032642
- gold: `فلماكشفناعنهمالرجزالياجلهمبالغوهاذاهمينكثون`
- pred: `االماكسنعمالجيذكلااجلنمبالمهاذاهمنينكصون`
- char_accuracy: 0.605

### hf_quran_md_ayah_route_minshawy_mujawwad_007_140_032792
- gold: `قالاغيراللهابغيكمالهاوهوفضلكمعليالعالمين`
- pred: `قالاضيومهالذيكمالاهنموفضلكمعللاالممين`
- char_accuracy: 0.625

### hf_quran_md_ayah_route_minshawy_mujawwad_007_170_033692
- gold: `والذينيمسكونبالكتابواقامواالصلاةانالانضيعاجرالمصلحين`
- pred: `ولمنيمسقنبييكتابواقاهاصلااكننالالضالرلصلحخين`
- char_accuracy: 0.596

### hf_quran_md_ayah_route_minshawy_mujawwad_007_174_033812
- gold: `وكذلكنفصلالاياتولعلهميرجعون`
- pred: `ووكذالكنفصلوااةوالعلههمممييرهجرون`
- char_accuracy: 0.519

### hf_quran_md_ayah_route_minshawy_mujawwad_007_181_034022
- gold: `وممنخلقناامةيهدونبالحقوبهيعدلون`
- pred: `ومنخالقوناهممكميفدونبلحتوابيهيعبيلون`
- char_accuracy: 0.548

### hf_quran_md_ayah_route_minshawy_mujawwad_007_199_034562
- gold: `خذالعفووامربالعرفواعرضعنالجاهلين`
- pred: `قوذللعقوااموبارفيواارضاناجنلين`
- char_accuracy: 0.562

### hf_quran_md_ayah_route_minshawy_mujawwad_008_015_035222
- gold: `ياايهاالذينامنوااذالقيتمالذينكفروازحفافلاتولوهمالادبار`
- pred: `ههولمنامنااملقيتمنذينكفاوزفمفلاتونولذبررن`
- char_accuracy: 0.537

### hf_quran_md_ayah_route_minshawy_mujawwad_008_018_035312
- gold: `ذلكمواناللهموهنكيدالكافرين`
- pred: `ذلكمموعناللوونكجلكافرين`
- char_accuracy: 0.692

### hf_quran_md_ayah_route_abdullah_basfar_010_084_043413
- gold: `وقالموسيياقومانكنتمامنتمباللهفعليهتوكلواانكنتممسلمين`
- pred: `وقالمووسياقومانكنتماامنتنبالهفعليهتوكلومعليهتورككلونانكنتمموالسلمين`
- char_accuracy: 0.615

### hf_quran_md_ayah_route_husary_mujawwad_013_029_052054
- gold: `الذينامنواوعملواالصالحاتطوبيلهموحسنماب`
- pred: `االذيناامنوعيلصالحاتطقوبالهموحسنمااب`
- char_accuracy: 0.684

### hf_quran_md_ayah_route_husary_mujawwad_014_041_053704
- gold: `ربنااغفرليولوالديوللمؤمنينيوميقومالحساب`
- pred: `قربنوخفاليولولديلالمؤمننينايماياقماحساب`
- char_accuracy: 0.615

### hf_quran_md_ayah_route_banna_019_001_067505
- gold: `كهيعص`
- pred: `ملاعياص`
- char_accuracy: 0.000

### hf_quran_md_ayah_route_banna_020_027_071225
- gold: `واحللعقدةمنلساني`
- pred: `وحلنلقدتمملسانن`
- char_accuracy: 0.625

### hf_quran_md_ayah_route_warsh_yassin_021_001_074496
- gold: `اقتربللناسحسابهموهمفيغفلةمعرضون`
- pred: `بسملهرومانحنقتربلناسحسابهموهمثيوفلتمعرضون`
- char_accuracy: 0.516

### hf_quran_md_ayah_route_warsh_yassin_025_001_085656
- gold: `تباركالذينزلالفرقانعليعبدهليكونللعالميننذيرا`
- pred: `تلانارحمانحيتبوكالذينلذلالفرقانعلاعلدهيليكونااللعالميننذير`
- char_accuracy: 0.545

### hf_quran_md_ayah_route_abdul_basit_murattal_026_026_088717
- gold: `قالربكموربابائكمالاولين`
- pred: `قالرضكموربضااباائكوااوونين`
- char_accuracy: 0.652

### hf_quran_md_ayah_route_abdul_basit_murattal_026_128_091777
- gold: `اتبنونبكلريعايةتعبثون`
- pred: `اتذنونبكاللريياايتنتعبثون`
- char_accuracy: 0.667

### hf_quran_md_ayah_route_abdul_basit_murattal_026_220_094537
- gold: `انههوالسميعالعليم`
- pred: `انههعسبعالعلم`
- char_accuracy: 0.647
