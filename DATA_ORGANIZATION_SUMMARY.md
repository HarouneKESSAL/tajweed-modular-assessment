# Tajweed Modular Assessment: Data Organization and Structure

A comprehensive guide to the data sources, processing pipeline, and corpus structure used in the project.

---

## 1. Data Sources and Datasets

### 1.1 Primary Audio Dataset: RetaSy

**Dataset:** Retasy Quranic audio dataset  
**Source:** HuggingFace (RetaSy/quranic_audio_dataset)  
**Type:** Crowdsourced learner recitations  
**Coverage:**
- ~7,000 total recitations
- 1,287 participants
- 1,166 annotated recitations (with human judgments on 6 categories)
- Non-Arabic speaker recitations (ideal for learner-error analysis)

**Main Manifest:**
- `data/manifests/retasy_train.jsonl`: **6,828 samples** - core training dataset

**Why RetaSy:**
- Provides real learner errors and mistakes
- Suitable for training error detection models
- Includes metadata: reciter ID, recording duration, surah/ayah information
- Each sample includes: audio path, canonical text, duration_ms, metadata

### 1.2 Reference Datasets: Quran-MD

**Purpose:** Large-scale clean recitation data for pretraining and weak-label experiments  
**Sources:** HuggingFace datasets

**Quran-MD Ayahs (Full verses):**
- `hf_quran_md_ayahs_unique48_r2.jsonl`: **96 samples** - unique 48-verse dataset with 2x redundancy
- Used for: All-ayah coverage experiments, routing validation
- Contains: Full-verse audio, exact text, reciter metadata

**Quran-MD Words (Word-level):**
- `hf_quran_md_words_pilot512_r4.jsonl`: **512 samples** - pilot subset
- `hf_quran_md_words_pilot5000_r8.jsonl`: **5,000 samples** - larger pilot
- Used for: Word-level content recognition, transfer learning experiments
- Contains: Individual word audio clips with text and position metadata

### 1.3 Reference Data: Quran JSON & Colored Mushaf

**Location:** `external/quranjson-tajwid/`  
**Purpose:** Structural Quran reference with Tajweed rule annotations  
**Format:** JSON files organized by surah number

**Extracted Manifest:**
- `data/manifests/quranjson_rules.jsonl`: **6,348 samples** - extracted rule information

**What It Provides:**
- Complete Quran text structure (all 114 surahs)
- Surah/verse/word boundaries
- Tajweed rule positions and types encoded as color spans in the colored Mushaf
- Canonical reference text for normalization
- Rule targets: madd, ghunnah, ikhfa, idgham, qalqalah, and other rules

**Critical Role in Data Pipeline:**
- Maps each RecASy sample to a surah and ayah in the standard Quran
- Defines expected Tajweed rule positions
- Provides weak labels (reference-based) when human annotations are unavailable
- Enables routing: indicates which rule modules should be applied to each segment
- Example: If quranjson indicates verse_3_1 contains a madd position, then audio samples for that verse are routed to the duration module

### 1.4 Enriched Combined Manifest

**Manifest:** `data/manifests/retasy_quranjson_train.jsonl` - **6,828 samples**

**Structure:**
- Joins RetaSy audio samples with Quran JSON reference data
- Each sample includes:
  - Audio metadata (path, duration, sample rate)
  - Normalized Arabic text
  - Surah/ayah identifiers
  - Tajweed rule targets (from colored Mushaf reference)
  - Routing flags (use_duration, use_transition, use_burst, use_content)
  - Reciter information
  - Trust level (golden=true for human-annotated, false for reference-based)

---

## 2. Data Organization: Folder Structure

### 2.1 `data/raw/`
**Status:** `.gitkeep` only (audio files not committed)  
**Purpose:** Placeholder for raw audio files

**Expected Structure:**
- `data/raw/retasy_train_audio/`: RetaSy WAV files referenced by manifests
- Named as: `000000_al_falaq.wav`, `000001_al_faatihah.wav`, etc.
- Each file has a corresponding entry in `retasy_train.jsonl`

### 2.2 `data/processed/`
**Status:** `.gitkeep` only (processed features not committed)  
**Purpose:** Extracted acoustic features (MFCC, wav2vec representations)

**Expected Contents:**
- MFCC features as `.pt` (PyTorch tensor) files: `sample_id.pt`
- wav2vec embeddings for content module
- Features are loaded on-the-fly during training from audio (when `use_toy_data=false`)

### 2.3 `data/alignment/` - **Alignment and Time Projections**
**Purpose:** Maps Tajweed rule positions to approximate time ranges in audio

**Content Duration Alignment:**
- `content_time_projection_pilot.jsonl` - **40 samples**
- `content_time_projection_correct200.jsonl` - **126 samples**
- `content_time_projection_scaled250.jsonl` - **250 samples**
- Maps: character → time (start_sec, end_sec)
- Method: Forced alignment techniques
- Used by: Content module for chunking and span evaluation

**Duration Module Alignments:**
- `duration_time_projection_strict.jsonl` - **983 samples**
- `duration_time_projection_preview.jsonl` - **1,435 samples**
- `duration_time_projection_preview_25.jsonl` - **1,435 samples (subset)**
- Maps: Madd/ghunnah rule position → time range
- Used by: Duration module, localized span detection

**Transition Module Alignments:**
- `transition_time_projection_strict.jsonl` - **983 samples**
- Maps: Ikhfa/idgham position → time range
- Used by: Transition module localized support

**Torchaudio Forced Alignment Output:**
- `torchaudio_forced_alignment_strict.jsonl` - **973 samples**
- `torchaudio_forced_alignment_preview.jsonl` - **1,435 samples**
- `content_torchaudio_forced_alignment_*.jsonl` - Content-specific versions
- Raw alignment output from wav2vec-based forced alignment

### 2.4 `data/manifests/` - **Task-Specific Training Corpora**

#### Core Training Manifests:

| Manifest | Samples | Purpose |
|----------|---------|---------|
| `retasy_train.jsonl` | 6,828 | Main training dataset (full RetaSy) |
| `retasy_quranjson_train.jsonl` | 6,828 | Enriched with Quran JSON rule metadata |
| `quranjson_rules.jsonl` | 6,348 | Extracted rule information from Quran JSON |

#### Content Module Manifests:

| Manifest | Samples | Purpose | Notes |
|----------|---------|---------|-------|
| `retasy_content_chunks.jsonl` | 1,944 | Chunked content dataset | Verses split into word/character chunks |
| `retasy_content_chunks_with_subchunks.jsonl` | 8,519 | Hierarchical chunks | Contains subchunk boundaries |
| `retasy_content_subchunks_textsplit_trainonly.jsonl` | 6,764 | Text-split validation | For text-unseen generalization |
| `retasy_content_chunks_textsplit_trainonly.jsonl` | 1,527 | Chunk variant text-split | Training-only version |
| `retasy_content_chunks_alignment_correct200.jsonl` | 309 | Alignment corpus (strict) | 200 verses with forced alignment |
| `retasy_content_chunks_alignment_pilot.jsonl` | 84 | Alignment corpus (pilot) | 40 verses for testing |
| `retasy_content_chunks_alignment_scaled250.jsonl` | 749 | Alignment corpus (medium) | 250 verses |
| `retasy_content_alignment_corpus_correct200.jsonl` | 126 | Full ayah alignment | Correct200 subset |
| `retasy_content_alignment_corpus_pilot.jsonl` | 40 | Full ayah alignment (pilot) | Small pilot version |
| `retasy_content_alignment_corpus_scaled250.jsonl` | 250 | Full ayah alignment (scaled) | Medium version |
| `retasy_content_chunks_curriculum_correct200_r10.jsonl` | 15,536 | Curriculum learning (large) | 10x repetition with curriculum ordering |
| `retasy_content_chunks_curriculum_correct200_r5.jsonl` | 7,901 | Curriculum learning (medium) | 5x repetition |
| `retasy_content_chunks_train_plus_alignment_*.jsonl` | 1,793-2,230 | Combined training+alignment | Different size variants |

**Content Chunk Structure:**
```json
{
  "id": "retasy_train_000020_chunk_00",
  "parent_id": "retasy_train_000020",
  "audio_path": "data\\raw\\retasy_train_audio\\...",
  "surah_name": "Al-Faatihah",
  "normalized_text": "الرَّحْمَٰن",
  "start_sec": 0.150,
  "end_sec": 1.735,
  "char_count": 6,
  "word_count": 1,
  "word_spans": [...],
  "quranjson_verse_key": "verse_3",
  "reciter_id": "Unknown"
}
```

#### Duration Module Manifests:

| Manifest | Samples | Purpose |
|----------|---------|---------|
| `retasy_duration_subset.jsonl` | 1,435 | Base duration training dataset |
| `retasy_duration_subset_coarse.jsonl` | 1,435 | Coarse-grained version |
| `retasy_duration_alignment_strict.jsonl` | 983 | Strict alignment (best quality) |
| `retasy_duration_alignment_weak.jsonl` | 452 | Weak labels (reference-based) |
| `retasy_duration_alignment_prep.jsonl` | 1,435 | Preprocessed version |
| `retasy_duration_alignment_corpus_torchaudio_strict.jsonl` | 973 | Forced-aligned duration corpus |
| `retasy_madd_subtype_subset.jsonl` | 1,388 | Madd subtypes (alif, ya, etc.) |

**Duration Labeling:**
- Labels: `none`, `madd`, `ghunnah`
- Madd subtypes: Long vowel (alif, ya, wa)
- Ghunnah: Nasalization rule
- Stored as: Per-sample rule labels + position spans

#### Transition Module Manifests:

| Manifest | Samples | Purpose |
|----------|---------|---------|
| `retasy_transition_subset.jsonl` | 690 | Base transition training dataset |

**Transition Labels:**
- `ikhfa` (concealment before 15 letters)
- `idgham` (assimilation into ya, ra, meem, noon)
- Multi-label: A sample can have both rules

#### Burst Module Manifests:

| Manifest | Samples | Purpose |
|----------|---------|---------|
| `retasy_burst_subset.jsonl` | 1,597 | Qalqalah/burst training dataset |

**Qalqalah Labeling:**
- Binary classification: qalqalah present/absent
- Letters: Qaf, Taa, Ba, Jeem (when in pause or doubled)

#### HuggingFace Quran Data Experiments:

| Manifest | Samples | Purpose |
|----------|---------|---------|
| `hf_quran_md_ayahs_unique48_r2.jsonl` | 96 | All unique ayahs from Quran-MD |
| `hf_quran_md_ayahs_unique48_r2_approx_chunks.jsonl` | 678 | Chunked version |
| `hf_quran_md_words_pilot512_r4.jsonl` | 512 | Word-level pilot |
| `hf_quran_md_words_pilot5000_r8.jsonl` | 5,000 | Word-level larger |

#### Curriculum Learning Variants:

| Manifest | Samples | Purpose |
|----------|---------|---------|
| `content_v6a_short_hf_ayah_r1_stage1_anchor.jsonl` | 5,000 | Stage 1: Anchor examples |
| `content_v6a_short_hf_ayah_r1_stage2_ayah_intro.jsonl` | 5,700 | Stage 2: Introduction to ayahs |
| `content_v6a_short_hf_ayah_r1_stage3_ayah_focus.jsonl` | 6,800 | Stage 3: Focus on ayahs |
| `content_v6a_short_hf_ayah_r1_vocab_all.jsonl` | 9,988 | Complete vocabulary coverage |

### 2.5 `data/analysis/` - **Evaluation and Experiment Results**

**Purpose:** Store evaluation outputs, confusion matrices, hardcases, and ablation comparisons

**Dataset Statistics and Analysis:**
- `hf_quran_md_ayahs_unique48_r2_summary.json` - Data distribution for Quran-MD
- `hf_quran_md_words_pilot5000_r8_summary.json` - Word dataset statistics
- `hf_quran_md_content_pilot_comparison.json` - Content experiment comparisons

**Content Module Results:**
- `chunked_content_*.json` - Experiments with chunked content model (100+ variants)
- `content_ayah_*.json` - Full ayah content recognition results
- `content_expansion_*.json` - Scaling experiments
- `content_lexicon_*.json` - Lexicon-constrained decoding results
- `content_whisper_*.json` - Whisper ASR baseline comparisons

**Duration Module Results:**
- `duration_fusion_*.json` - Learned fusion calibrator experiments
- `duration_confusion*.json` - Error analysis and confusion matrices
- `duration_hardcases*.json` - Difficult example analysis

**Transition Module Results:**
- `transition_confusions*.json` - Transition error analysis
- `transition_hardcases*.json` - Difficult transition cases
- `transition_multilabel_*.json` - Multi-label transition experiments

**System-Level Results:**
- `modular_suite_*.json` - Full pipeline evaluations (30+ variants)
- `final_baseline_results.json` - Official baseline snapshot
- `whole_system_status_report_v2.json` - System performance overview

**Thesis-Oriented Results:**
- `thesis_ablation_v2/THESIS_ABLATION_SUMMARY.md` - Final ablation summary
- `thesis_ablation_v2/MODULE_INTERNAL_ABLATION_REPORT.md` - Module-level ablations

### 2.6 `data/external/` - **External Reference Data**
**Status:** Empty (`.gitkeep`)  
**Purpose:** External datasets and references  
**Note:** Large reference files are typically fetched at runtime from HuggingFace

### 2.7 `data/interim/` - **Intermediate Processing**
**Status:** Empty (`.gitkeep`)  
**Purpose:** Intermediate artifacts during data pipeline processing

---

## 3. Data Processing Pipeline

### 3.1 High-Level Pipeline Overview

```
Raw Audio                    External References
    ↓                               ↓
   RetaSy                    Quran JSON / Colored Mushaf
    ↓                               ↓
    └─────────────────┬────────────┘
                      ↓
          Normalization + Alignment
                      ↓
          Enriched Combined Manifest
          (retasy_quranjson_train.jsonl)
                      ↓
          ┌───────────────────────────┐
          ↓           ↓           ↓    ↓
        Content   Duration   Transition  Burst
        Chunks    Subsets    Subsets    Subsets
          ↓           ↓           ↓        ↓
       Training Manifests for Each Module
```

### 3.2 Data Processing Scripts

**Location:** `scripts/data/`

#### 1. **build_manifests.py**
- **Purpose:** Create baseline manifest structure
- **Input:** Raw audio paths, metadata
- **Output:** Initial JSONL manifest
- **Key Classes:**
  - `ManifestEntry`: Single audio sample record
  - `save_manifest()`: Write manifest to JSONL

#### 2. **build_torchaudio_alignment_corpus.py**
- **Purpose:** Run forced alignment on audio
- **Input:** Manifest file + audio files
- **Process:**
  - Uses `torchaudio` forced alignment
  - Maps characters to time ranges
  - Uses `uroman` for phonetic transliteration
  - Sanitizes Arabic text (removes diacritics, special chars)
- **Output:** Alignment manifest with (start_sec, end_sec) for each character
- **Key Functions:**
  - `sanitize_alignment_text()`: Clean Arabic text
  - `run_uroman()`: Convert text to romanization for alignment
  - `audio_duration_sec()`: Get audio length

#### 3. **extract_features.py**
- **Purpose:** Precompute acoustic features
- **Input:** Manifest + audio paths
- **Process:**
  - Reads audio from manifest entries
  - Extracts MFCC features (configurable in data.yaml)
  - Saves as PyTorch `.pt` files
  - Updates manifest with feature_path
- **Output:** Feature tensors + updated manifest

#### 4. **import_hf_quran_content.py**
- **Purpose:** Import Quran-MD datasets from HuggingFace
- **Datasets:**
  - `Buraaq/quran-md-ayahs` (full verses)
  - `Buraaq/quran-md-words` (word-level)
  - `rabah2026/Quran-Ayah-Corpus` (alternate corpus)
- **Process:**
  - Loads dataset from HuggingFace
  - Extracts audio and metadata
  - Normalizes Arabic text (removes diacritics, standardizes letter variants)
  - Saves audio as WAV files
  - Creates manifest entries
- **Key Functions:**
  - `normalize_arabic_text()`: Standardize Arabic (alef, ya, ta variants)
  - `audio_duration_sec()`: Get duration from soundfile
  - `save_audio_without_torchcodec()`: Handle audio conversion

#### 5. **build_hf_quran_md_ayah_transition_pilot.py**
- **Purpose:** Create transition labels for Quran-MD ayahs
- **Process:**
  - Loads ayah dataset
  - Detects ikhfa and idgham patterns in text
  - Generates multi-hot labels: [ikhfa_present, idgham_present]
- **Key Functions:**
  - `transition_labels_from_text()`: Extract transition rules from Arabic
  - `has_noon_pattern()`: Check for noon before trigger letters
  - `IKHFA_TRIGGER_LETTERS`: {ت ث ج د ذ ز س ش ص ض ط ظ ف ق ك}
  - `IDGHAM_TRIGGER_LETTERS`: {ي ر م ل و ن}

#### 6. **audit_clean_expansion_pool.py**
- **Purpose:** Validate and filter clean data
- **Process:**
  - Audits Quran-MD corpus for quality
  - Checks text normalization
  - Filters problematic samples
- **Output:** Validated clean dataset manifest

### 3.3 Text Normalization Pipeline

**Why Important:** Arabic can be written with diacritics, spelling variants, and orthographic markers

**Normalization Steps:**

```python
def normalize_arabic(text: str) -> str:
    # 1. Remove diacritical marks (harakat)
    text = remove_diacritics(text)  # ـٌٍَُِّْـ
    
    # 2. Standardize alef variants → alef
    text = text.replace('أ', 'ا')    # hamza-alef
    text = text.replace('إ', 'ا')    # alef-with-below
    text = text.replace('آ', 'ا')    # alef-with-above
    text = text.replace('ٱ', 'ا')    # alef-wasla
    
    # 3. Standardize ya variants → ya
    text = text.replace('ى', 'ي')    # alef-maksura
    
    # 4. Remove tatweel (lengthening mark)
    text = text.replace('ـ', '')
    
    # 5. Normalize spaces
    text = ' '.join(text.split())
    
    return text
```

**Applied To:**
- Quran JSON verse text
- RetaSy recitation text
- All content recognition targets

### 3.4 Routing and Module-Specific Extraction

**Routing Logic:**
Once the enriched manifest (retasy_quranjson_train.jsonl) is created, samples are routed to modules:

```python
for each sample in retasy_quranjson_train.jsonl:
    
    # Extract rule targets from Quran JSON reference
    rules = extract_rules_from_colored_mushaf(surah_num, verse_num)
    
    # Content module: always present
    if has_canonical_text:
        content_corpus.append(sample)
        
    # Duration module: if madd or ghunnah present
    if 'madd' in rules or 'ghunnah' in rules:
        duration_corpus.append(sample)
    
    # Transition module: if ikhfa or idgham present
    if 'ikhfa' in rules or 'idgham' in rules:
        transition_corpus.append(sample)
    
    # Burst module: if qalqalah present
    if 'qalqalah' in rules:
        burst_corpus.append(sample)
```

**Result:** Module-specific manifests like:
- `retasy_duration_subset.jsonl` (1,435 samples)
- `retasy_transition_subset.jsonl` (690 samples)
- `retasy_burst_subset.jsonl` (1,597 samples)

---

## 4. Content Module Data

### 4.1 Content Recognition Strategy

**Problem:** Verify whether the learner recited the correct Arabic text  
**Approach:** CTC-based character-level speech recognition

### 4.2 Chunking Strategy (Evolution)

**v1: Full Verse** (Original, abandoned)
- Processed entire verses (10-50 words)
- Problem: Long sequences collapse with CTC
- Result: High deletion bias, poor performance

**v2: Chunked Content** (Current)
- Split verses into word or multi-word chunks
- Typical chunk: 1-5 words, 1-2 seconds
- Chunks maintain semantic meaning
- Features: Start/end time, character count, word count

**v3: Sub-chunks** (Experimental)
- Hierarchical: verse → chunk → sub-chunk
- Enables multi-level evaluation
- Not in official baseline

**Data Generation:**

```
Input: verse text + alignment (char → time)
Process:
  1. Load normalized verse text
  2. Get character-to-time mapping from alignment corpus
  3. Split on word boundaries (spaces in normalized text)
  4. For each word/chunk:
     - Extract text
     - Get start_sec, end_sec from character timing
     - Count characters and words
     - Create chunk ID: "parent_id_chunk_XX"
Output: Chunk manifest entry
```

### 4.3 Content Dataset Variants

| Variant | Samples | Purpose |
|---------|---------|---------|
| `retasy_content_chunks.jsonl` | 1,944 | Base chunked dataset |
| Text-split versions | 1,527-6,764 | Text-unseen generalization testing |
| Curriculum versions | 7,901-15,536 | Multi-epoch curriculum learning |
| Alignment corpus | 126-749 | Forced-alignment training |
| Sub-chunk versions | 6,764-8,519 | Hierarchical structure |

### 4.4 Content Alignment Data

**Sources:**
- `data/alignment/content_time_projection_*.jsonl`: Character-level time alignment
- `data/alignment/content_torchaudio_forced_alignment_*.jsonl`: Forced alignment output

**Structure:**
```json
{
  "id": "sample_id",
  "audio_path": "...",
  "text": "normalized_text",
  "character_alignment": [
    {"char": "م", "start_sec": 0.0, "end_sec": 0.15},
    {"char": "ا", "start_sec": 0.15, "end_sec": 0.30},
    ...
  ]
}
```

### 4.5 Content Decoder Configurations

**Location:** `checkpoints/`

**Closed-Set Decoder:**
- `content_chunked_decoder.json`: Known-verse content decoding
- Constrained to phrase list vocabulary
- Used for verse-specific assessment

**Open Decoder:**
- `content_chunked_decoder_open.json`: General vocabulary
- `content_chunked_decoder_open_hd96.json`: Improved version
- Not constrained to known verses
- Better generalization

**Configuration Parameters:**
```yaml
decoder_type: "lexicon_constrained"  # or "greedy"
blank_penalty: 1.6                   # Reduces over-deletion
lexicon_path: "path/to/vocab.txt"    # For constrained decoding
nbest: 1
```

---

## 5. Tajweed Modules Data

### 5.1 Duration Module Data

**Purpose:** Detect madd (vowel lengthening) and ghunnah (nasalization)

**Labels:**
- `none`: No duration rule
- `madd`: Long vowel (2-6 haraka)
- `ghunnah`: Nasal n/m with sound continuation

**Duration Subsets:**
- `retasy_duration_subset.jsonl`: Base dataset (1,435 samples)
- `retasy_duration_alignment_strict.jsonl`: Forced-aligned (983 samples)
- `retasy_duration_alignment_weak.jsonl`: Reference labels (452 samples)
- `retasy_duration_subset_coarse.jsonl`: Coarse-grained (1,435 samples)

**Time Projection Data:**
- Maps duration rule position → (start_sec, end_sec) in audio
- Sources:
  - `data/alignment/duration_time_projection_strict.jsonl`: (983 samples)
  - `data/alignment/duration_time_projection_preview.jsonl`: (1,435 samples)

**Madd Subtypes:**
- `retasy_madd_subtype_subset.jsonl` (1,388 samples)
- Subtypes: long-vowel-alif, long-vowel-ya, long-vowel-wa
- Used for detailed duration analysis

### 5.2 Transition Module Data

**Purpose:** Detect assimilation (idgham) and concealment (ikhfa)

**Labels:**
- `none`: No transition rule
- `ikhfa`: Concealment (noon before 15 letters: ت ث ج د ذ ز س ش ص ض ط ظ ف ق ك)
- `idgham`: Assimilation (noon before: ي ر م ل و ن)
- Multi-label: Sample can have both ikhfa and idgham

**Transition Dataset:**
- `retasy_transition_subset.jsonl` (690 samples)
- Extracted from samples where transition rules exist
- Includes position spans and timing

**Time Projection Data:**
- `data/alignment/transition_time_projection_strict.jsonl` (983 samples)
- Maps transition rule position to audio time range

### 5.3 Burst Module Data

**Purpose:** Detect qalqalah (burst articulation of emphatic letters)

**Labels:**
- Binary: qalqalah present/absent
- Letter rule: Applies to Qaf, Taa, Ba, Jeem when in pause position or doubled

**Burst Dataset:**
- `retasy_burst_subset.jsonl` (1,597 samples)
- Includes: audio path, verse text, rule positions, timing

---

## 6. Data Configuration Files

### 6.1 `configs/data.yaml`

**Audio Processing Parameters:**

```yaml
# Sample rate for audio loading
sample_rate: 16000

# MFCC feature extraction
n_mfcc: 13

# Audio feature computation
num_workers: 0

# Toy data mode (for quick testing)
use_toy_data: true

# Speed normalization (simulates accent variations)
normalize_speed: true
target_speech_rate: 12.0  # words per second (Quran standard)
min_speed_factor: 1.0
max_speed_factor: 1.35
```

**Usage in Training:**
- Loads audio at 16kHz sample rate
- Computes 13 MFCCs + delta features (39-dim total)
- Normalizes audio speed to target rate
- Adds randomized speed variation (1.0-1.35x) for data augmentation

### 6.2 `configs/error_weights.yaml`

**Error Severity Weighting System**

```yaml
version: 1
scale: 3.0  # Global severity multiplier

categories:
  content:
    wrong_word:
      weight: 10.0        # Most critical
      severity: critical
      lahn_type: jali     # apparent error
    
    missing_word:
      weight: 10.0
      severity: critical
      lahn_type: jali
    
    extra_word:
      weight: 8.0
      severity: major
      lahn_type: jali
    
    letter_substitution:
      weight: 7.0
      severity: major
      lahn_type: jali
    
    minor_text_normalization_difference:
      weight: 1.0
      severity: minor
      lahn_type: technical

  duration:
    severe_madd_error:
      weight: 4.0
      severity: medium
      lahn_type: khafi    # hidden error
    
    minor_madd_duration_error:
      weight: 2.0
      severity: minor
      lahn_type: khafi
    
    ghunnah_duration_error:
      weight: 3.0
      severity: medium
      lahn_type: khafi

  transition:
    wrong_transition_rule:
      weight: 4.0
      severity: medium
      lahn_type: khafi
```

**Purpose:**
- Not neural attention; semantic error prioritization
- Content errors weighted higher than timing errors
- Guides feedback generation and scoring aggregation
- Used in: `weighted_penalty = error_count * severity_weight * confidence`

### 6.3 `configs/train.yaml`

**Training Hyperparameters:**

```yaml
# Random seed for reproducibility
seed: 7

# Training epochs
epochs: 20

# Batch size
batch_size: 4

# Learning rate
learning_rate: 0.001

# Device (cpu or cuda)
device: cpu

# Checkpoint save directory
checkpoint_dir: checkpoints
```

**Note:** Batch size of 4 with multi-worker audio loading suggests limited GPU memory or development setup.

### 6.4 `configs/model_content.yaml`

**Content Module Architecture:**

```yaml
# Pretrained SSL model
ssl_model_name: WAV2VEC2_BASE
hidden_dim: 64              # LSTM hidden dimension
dropout: 0.1
num_phonemes: 11            # Arabic phoneme set
```

**Model Stack:**
1. WAV2VEC2 feature extractor → 768-dim representations
2. Projection → hidden_dim (64)
3. BiLSTM → context encoding
4. CTC head → phoneme probabilities

### 6.5 `configs/model_duration.yaml`

**Duration Module Architecture:**

```yaml
input_dim: 39               # MFCC + delta + delta-delta
hidden_dim: 32
num_layers: 1
dropout: 0.1
num_phonemes: 11
num_rules: 6                # none, madd, ghunnah, ikhfa, idgham, qalqalah

# Multi-task learning weights
lambda_ctc: 1.0             # CTC loss weight
lambda_rule: 0.7            # Rule classification loss weight

# Score combination in decoding
phoneme_score_weight: 0.7   # 70% phoneme-aware
rule_score_weight: 0.3      # 30% rule direct

# Class imbalance handling
rule_class_weights:
  none: 1.0
  madd: 1.0
  ghunnah: 4.0    # Upweighted (harder to detect)
  ikhfa: 1.0
  idgham: 1.0
  qalqalah: 1.0
```

### 6.6 `configs/model_transition.yaml`

**Transition Module Architecture:**

```yaml
# Input features
mfcc_dim: 39                # MFCC features
ssl_dim: 64                 # SSL features (optional)
hidden_dim: 64
num_layers: 1
dropout: 0.1
num_rules: 3                # none, ikhfa, idgham (+ combinations)
```

### 6.7 `configs/model_burst.yaml`

**Burst Module Architecture:**

```yaml
input_dim: 39
channels: [16, 32]          # CNN channels
dropout: 0.1
num_classes: 2              # Binary: qalqalah / no-qalqalah
```

---

## 7. Training Manifests and Data Splits

### 7.1 Train/Validation/Test Splits

**Approach:** Text-based and reciter-based splits for generalization testing

**Text-Split Variants:**
- `retasy_content_chunks_textsplit_trainonly.jsonl`: Training-only version
- `retasy_content_chunks_alignment_*_no_textsplit_val.jsonl`: Validation with text overlap

**Purpose:**
- Text-split: Unseen text generalization (lyrics not in training)
- Text-overlap: Seen text, different reciter generalization

**Reciter-Split Variants:**
- `content_open_recitersplit.jsonl`: Different reciter evaluation
- `*_recitersplit_*.jsonl`: Reciter-based splits

### 7.2 Clean Data Pretraining

**HuggingFace Quran-MD Pretraining Sets:**

| Manifest | Samples | Training Strategy |
|----------|---------|-------------------|
| `hf_quran_md_words_pilot512_r4_textsplit_hd96_train.json` | (train+val) | Word-level pretraining |
| `hf_quran_md_ayahs_unique48_r2_recitersplit_hd96_train.json` | (train+val) | Ayah-level pretraining |

**Curriculum Learning:**

Manifests named with stages indicate progressive difficulty:

```
content_v6a_short_hf_ayah_r1_stage1_anchor.jsonl      (5,000)  → Basic examples
content_v6a_short_hf_ayah_r1_stage2_ayah_intro.jsonl  (5,700)  → Introduce ayahs
content_v6a_short_hf_ayah_r1_stage3_ayah_focus.jsonl  (6,800)  → Focus on ayahs
content_v6a_short_hf_ayah_r1_vocab_all.jsonl          (9,988)  → Full vocabulary
```

Training order: stage1 → stage2 → stage3 → vocab_all

---

## 8. Data Statistics Summary

### 8.1 Dataset Sizes

| Category | Dataset | Samples | Purpose |
|----------|---------|---------|---------|
| **Main Training** | retasy_train | 6,828 | Primary learner corpus |
| **Content** | retasy_content_chunks | 1,944 | Chunked content |
| **Content** | content_v6a_vocab_all | 9,988 | Curriculum final stage |
| **Content** | curriculum_r10 | 15,536 | Large curriculum (10x) |
| **Duration** | retasy_duration_subset | 1,435 | Duration labeling |
| **Duration** | duration_strict_aligned | 973 | Forced-aligned |
| **Transition** | retasy_transition_subset | 690 | Transition rules |
| **Burst** | retasy_burst_subset | 1,597 | Qalqalah detection |
| **Clean Reference** | Quran-MD ayahs | 96 | All Quran ayahs |
| **Clean Reference** | Quran-MD words | 5,000 | Word-level pretraining |
| **Reference** | quranjson_rules | 6,348 | Rule reference data |

### 8.2 Audio Duration

**RetaSy Dataset:**
- Sample rate: 16,000 Hz
- Typical sample duration: 6-10 seconds (verse-level)
- Chunk duration: 0.5-2 seconds (word-level)
- Total duration: ~10-15 hours of audio

### 8.3 Text/Content Statistics

**Normalized Arabic:**
- Alphabet: 28 base letters + diacritics (removed before training)
- CTC phoneme set: 11 phonemes
- Vocabulary size (chunks): 678-9,988 unique chunks depending on variant

---

## 9. Data Quality and Validation

### 9.1 Quality Tiers

**Golden/Trusted Data:**
- Samples with `golden=true` in manifest
- Human annotated Tajweed judgments
- Count: ~1,166 from RetaSy (reported in paper)

**Reference-Based Labels:**
- Derived from Quran JSON / colored Mushaf
- Used when human annotations unavailable
- Assumption: Canonical rules apply to learner sample
- Risk: Learner may pronounce rules differently than expected

**Weak Labels:**
- Estimated from weak learning signals
- Example: Alignment confidence scores
- Used in: `*_weak.jsonl` manifests

### 9.2 Validation Mechanisms

**Forced Alignment Quality:**
- Audio → text character alignment via wav2vec
- Used to filter unreliable alignments
- Confidence scoring for alignment quality
- Produces: `data/alignment/torchaudio_forced_alignment_*.jsonl`

**Text Normalization Validation:**
- Script: `audit_clean_expansion_pool.py`
- Checks: Diacritic removal, variant standardization, space consistency
- Produces: Validated manifests and audit reports

**Module-Specific Validation:**
- Duration: Confusion analysis (`duration_confusions.json`)
- Transition: Multi-label consistency checks
- Content: Character error rate validation

---

## 10. Data Access in Code

### 10.1 Loading Manifests in Training

```python
from tajweed_assessment.data.manifests import load_manifest

# Load JSONL manifest
entries = load_manifest(Path("data/manifests/retasy_train.jsonl"))

for entry in entries:
    audio_path = entry.audio_path
    text = entry.normalized_text
    rules = entry.rule_labels
    duration = entry.duration_ms
    # ... use in training dataset
```

### 10.2 Feature Extraction

```python
from tajweed_assessment.features.mfcc import extract_mfcc_features
import soundfile as sf

# Load audio and extract MFCC
audio, sr = sf.read(audio_path)  # 16kHz
mfcc_features = extract_mfcc_features(audio)  # Shape: (time, 39)
```

### 10.3 Alignment Data Access

```python
# Load alignment data
align_entries = load_manifest(
    Path("data/alignment/content_time_projection_strict.jsonl")
)

for entry in align_entries:
    char_alignment = entry.character_alignment
    # Example: [{"char": "م", "start_sec": 0.0, "end_sec": 0.15}, ...]
```

---

## 11. Summary: Data Flow from Raw to Training

```
1. RAW SOURCES
   ├─ RetaSy audio (6,828 learner recitations)
   ├─ Quran JSON reference (all surahs)
   └─ Quran-MD clean data (optional pretraining)

2. ENRICHMENT & ALIGNMENT
   ├─ Text normalization (remove diacritics, standardize letters)
   ├─ Forced alignment (character → time mapping)
   └─ Rule extraction (from colored Mushaf reference)

3. COMBINED MANIFEST
   └─ retasy_quranjson_train.jsonl (6,828 samples with all metadata)

4. MODULE-SPECIFIC ROUTING
   ├─ Content chunks (1,944-15,536 depending on variant)
   ├─ Duration subset (1,435 samples)
   ├─ Transition subset (690 samples)
   └─ Burst subset (1,597 samples)

5. DATA AUGMENTATION & VARIANTS
   ├─ Text-split validation (unseen text generalization)
   ├─ Reciter-split evaluation (unseen reciter)
   ├─ Curriculum stages (progressive difficulty)
   └─ Speed normalization (simulated accents: 1.0-1.35x)

6. FEATURE EXTRACTION (on-demand)
   ├─ MFCC features (39-dim: 13 coeff + delta + delta-delta)
   └─ Wav2Vec SSL features (768-dim representation)

7. TRAINING
   ├─ Content module: CTC on chunks, phoneme classification
   ├─ Duration module: Rule classification + CTC multitask
   ├─ Transition module: Rule classification on whole-verse
   └─ Burst module: Binary qalqalah classification
```

---

## 12. Key Data Insights

### 12.1 Strengths

✅ **Comprehensive Quran Reference:** Colored Mushaf provides complete rule position labels  
✅ **Learner-Oriented Primary Data:** RetaSy contains real errors, not just clean speech  
✅ **Modular Corpus Organization:** Clear separation by rule type  
✅ **Multi-View Evaluation:** Text-split, reciter-split, curriculum variants  
✅ **Alignment Infrastructure:** Forced alignment for fine-grained rule positioning  
✅ **Clean Data Available:** Quran-MD for pretraining and scalability  

### 12.2 Limitations

⚠️ **Golden Annotations Limited:** Only ~1,166 RetaSy samples with human Tajweed judgments  
⚠️ **Reference Labels as Proxy:** Most labels derived from Quran JSON, not human annotations  
⚠️ **Limited Learner Error Diversity:** ~1,287 learners may not cover all common mistakes  
⚠️ **Qalqalah Underrepresented:** Only 1,597 burst samples vs 6,828 total  
⚠️ **No Ground Truth Duration Measurements:** Duration rules inferred from rule presence, not actual measurements  

---

## 13. File Location Quick Reference

| Data Type | Location | Key Files |
|-----------|----------|-----------|
| Main Training Data | `data/manifests/` | `retasy_train.jsonl`, `retasy_quranjson_train.jsonl` |
| Content Chunks | `data/manifests/` | `retasy_content_chunks*.jsonl` (1,944-15,536 samples) |
| Duration Data | `data/manifests/` | `retasy_duration_subset.jsonl` (1,435 samples) |
| Transition Data | `data/manifests/` | `retasy_transition_subset.jsonl` (690 samples) |
| Burst Data | `data/manifests/` | `retasy_burst_subset.jsonl` (1,597 samples) |
| Alignments | `data/alignment/` | `*_time_projection_*.jsonl`, `*_forced_alignment_*.jsonl` |
| Rule Reference | `external/quranjson-tajwid/` | Quran JSON files organized by surah |
| Evaluation Results | `data/analysis/` | `*.json` experiment outputs, ablations, confusion matrices |
| Data Processing Scripts | `scripts/data/` | `build_manifests.py`, `build_torchaudio_alignment_corpus.py`, etc. |
| Data Config | `configs/` | `data.yaml`, `error_weights.yaml`, `model_*.yaml` |
| Decoders & Thresholds | `checkpoints/` | `content_chunked_decoder*.json`, `transition_thresholds.json`, etc. |

---

## 14. Conclusion

The tajweed-modular-assessment project uses a **structured, reference-driven data pipeline** that:

1. **Anchors to Quranic canonical text** through Quran JSON / colored Mushaf reference
2. **Enriches learner audio** with precise rule locations and routing information
3. **Separates by Tajweed rules** into dedicated module-specific manifests
4. **Validates with multiple splits** (text-based, reciter-based) for robust evaluation
5. **Supports iterative experimentation** through data variant manifests and ablation outputs
6. **Balances learner errors** (RetaSy) with clean reference data (Quran-MD) for robust training

This organization enables modular development, clear error analysis, and principled evaluation of specialized Tajweed rule detectors.
