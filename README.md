# Tajweed Modular Assessment

Système modulaire d'évaluation du Tajweed coranique. Analyse une récitation audio et évalue les règles de Tajweed avec des modules spécialisés (durée, transition, burst, contenu).

**Auteurs** : HADDAD Ahmed Ayoub & KESSAL Haroune  
**Encadrants** : Dr. CHENAIT Manel & Dr. BERKANI Lamia

---

## Sommaire

1. [Structure du projet](#1-structure-du-projet)
2. [Configuration obligatoire](#2-configuration-obligatoire)
3. [Fichiers de configuration](#3-fichiers-de-configuration)
4. [Installation](#4-installation)
5. [Lancement](#5-lancement)
6. [Modules principaux](#6-modules-principaux)
7. [Scripts](#7-scripts)
8. [Données](#8-données)
9. [Checkpoints](#9-checkpoints)
10. [Résultats clés](#10-résultats-clés)
11. [Remarques](#11-remarques)

---

## 1. Structure du projet

```
tajweed-modular-assessment/
│
├── README.md                      ← CE FICHIER
├── requirements.txt               ← Dépendances Python principales
├── requirements-whisper.txt       ← Dépendances optionnelles Whisper
├── .gitignore / .gitattributes    ← Config Git
│
├── app/                           ← API applicative (FastAPI)
│   ├── __init__.py
│   └── main.py                    ← Point d'entrée serveur
│
├── configs/                       ← Fichiers de configuration (25 fichiers)
│   ├── data.yaml                  ← Config audio/données : sample_rate, MFCC
│   ├── train.yaml                 ← Config entraînement : seed, batch_size, lr, epochs
│   ├── train_content_whisper_ctc.yaml ← Entraînement Whisper CTC
│   ├── model_duration.yaml        ← Architecture modèle durée (BiLSTM)
│   ├── model_transition.yaml      ← Architecture modèle transition (BiLSTM)
│   ├── model_burst.yaml           ← Architecture modèle burst (CNN)
│   ├── model_content.yaml         ← Architecture contenu legacy (wav2vec+CTC)
│   ├── model_content_hd96.yaml    ← Architecture contenu HD96 (principal)
│   ├── model_content_whisper.yaml ← Architecture Whisper
│   ├── model_content_whisper_ctc.yaml ← Whisper + CTC
│   ├── model_content_whisper_ctc_bilstm.yaml ← Whisper + CTC + BiLSTM
│   ├── error_weights.yaml         ← Poids des erreurs par module
│   ├── learned_router_thresholds.yaml ← Seuils routeur v1
│   ├── learned_router_v5_thresholds.yaml ← Seuils routeur v5 (principal)
│   ├── transition_multilabel_thresholds.yaml ← Seuils transition multi-label
│   ├── learned_routing_features_v1.json ← Features routage v1
│   ├── learned_routing_features_v2.json ← Features routage v2
│   ├── learned_routing_features_v3_group_text.json ← Features routage v3
│   ├── learned_routing_features_v4_rule_aware_group_text.json ← v4 rule-aware
│   ├── learned_routing_features_v5_retasy_hf_rule_aware_group_text.json ← v5
│   ├── content_ayah_decoder_bp12.json ← Décodeur ayah (blank penalty 1.2)
│   ├── content_chunked_decoder_beam_bp04.json ← Décodeur beam (bp 0.4)
│   ├── content_chunked_decoder_eval_lexicon_bp04.json ← Décodeur lexique eval
│   ├── production_content_gate.json ← Config production content gate
│   ├── whole_system_baseline_v2.json ← Baseline système complet v2
│   └── rule_manifest_json.json    ← Manifest des règles de Tajweed supportées
│
├── checkpoints/                   ← Poids des modèles entraînés (70 Go, ignoré par Git)
│   ├── duration_module.pt         ← Modèle de durée (99.27% accuracy)
│   ├── transition_module.pt       ← Modèle de transition (91.01% accuracy)
│   ├── burst_module.pt            ← Modèle burst/qalqalah (87.54% accuracy)
│   ├── content_chunked_module_hd96_reciter.pt ← Contenu principal
│   ├── duration_fusion_calibrator_approved.pt ← Calibrateur fusion durée
│   ├── learned_router_v5_retasy_hf_rule_aware_group_text.pt ← Routeur appris
│   ├── content_asr_whisper_medium_quran_v2_weighted/ ← Content gate (23 Go)
│   ├── content_asr_whisper_*/      ← Autres checkpoints Whisper
│   ├── content_chunked_module_*.pt ← Variantes contenu (~50 fichiers)
│   ├── localized_duration_model.pt ← Localisation temporelle durée
│   ├── localized_transition_model.pt ← Localisation temporelle transition
│   ├── learned_router_v*.pt        ← Routeurs appris (v1-v5)
│   ├── duration_fusion_calibrator*.pt ← Calibrateurs fusion
│   ├── content_multitask_*.pt      ← Modèles multi-tâches
│   ├── content_v6*.pt              ← Modèles contenu v6
│   ├── content_quran_md_*.pt       ← Modèles Quran-MD
│   ├── content_ayah_hf_*.pt        ← Modèles par ayah HuggingFace
│   └── *.json                      ← Configs décodeurs/seuils (10 fichiers)
│
├── data/                           ← Données du projet
│   ├── alignment/                  ← Alignements temporels (12 fichiers .jsonl)
│   ├── analysis/                   ← Résultats d'évaluation (200+ fichiers JSON/MD/CSV)
│   │   ├── thesis_ablation_v2/     ← Résumés d'ablation thèse
│   │   ├── ablations/              ← Sorties d'ablation
│   │   ├── modular_suite_*.json    ← Évaluations modulaires
│   │   └── whole_system_status_report_v2.md ← Rapport final
│   ├── external/                   ← Références externes
│   ├── interim/                    ← Données intermédiaires (ignoré par Git)
│   ├── manifests/                  ← Manifests d'entraînement (87 fichiers .jsonl)
│   │   ├── retasy_train.jsonl      ← Dataset principal Retasy
│   │   ├── retasy_duration_*.jsonl ← Manifests durée
│   │   ├── retasy_transition_*.jsonl ← Manifests transition
│   │   ├── retasy_burst_subset.jsonl ← Manifest burst
│   │   ├── retasy_content_chunks*.jsonl ← Manifests contenu
│   │   ├── learned_routing_dataset_v*.jsonl ← Datasets routage
│   │   ├── content_ayah_hf_*.jsonl ← Contenu par ayah
│   │   ├── content_v6*.jsonl       ← Manifests v6
│   │   ├── multitask_content_*.jsonl ← Manifests multi-tâches
│   │   ├── hf_quran_md_*.jsonl     ← Manifests Quran-MD
│   │   ├── quran_content_reference_full.jsonl ← Référence texte Quran
│   │   ├── quran_tajweed_reference_full.jsonl ← Référence règles Tajweed
│   │   └── quranjson_rules.jsonl   ← Règles extraites du Quran JSON
│   ├── processed/                  ← Données transformées (ignoré par Git)
│   ├── raw/                        ← Audio brut (ignoré par Git)
│   └── uploads/                    ← Uploads utilisateurs (ignoré par Git)
│
├── docs/                           ← Documentation supplémentaire
│   ├── architecture_notes.md       ← Notes d'architecture
│   ├── content_scoring_architecture.md ← Architecture scoring contenu
│   ├── experiment_plan.md          ← Plan d'expérimentation
│   ├── folder_roles.md             ← Rôles des dossiers
│   └── repo_tree.txt               ← Arborescence du dépôt
│
├── external/                       ← Références externes
│   └── quranjson-tajwid/           ← Données Quran/Tajweed de référence
│
├── figures/                        ← Figures générées
│   └── experiments/                ← Figures expérimentales
│
├── notebooks/                      ← Notebooks Jupyter d'exploration
│
├── papers/                         ← Documents PDF du mémoire
│   ├── Chapitre_Conception.pdf
│   ├── chapitre_etat_de_art.pdf
│   └── Conceptual_Framework.pdf
│
├── src/tajweed_assessment/         ← Package Python principal
│   ├── __init__.py
│   ├── settings.py                 ← ⚙️ Configuration centrale des chemins
│   ├── alignment/                  ← Alignement et projection temporelle
│   │   ├── __init__.py
│   │   ├── prep.py                 ← Préparation alignements
│   │   └── time_projection.py      ← Projection règles → temps
│   ├── data/                       ← Gestion des données
│   │   ├── audio.py                ← Chargement audio
│   │   ├── collate.py              ← Collation des batches
│   │   ├── dataset.py              ← Logique de dataset
│   │   ├── hf_retasy.py            ← Helper HuggingFace/Retasy
│   │   ├── labels.py               ← Gestion des labels
│   │   ├── localized_duration_dataset.py ← Dataset durée localisée
│   │   ├── localized_transition_dataset.py ← Dataset transition localisée
│   │   ├── manifests.py            ← Chargement manifests
│   │   ├── merge_manifest.py       ← Fusion de manifests
│   │   ├── quranjson_rules.py      ← Extraction règles Quran JSON
│   │   ├── real_duration_audio_dataset.py ← Dataset audio durée réelle
│   │   ├── real_duration_dataset.py ← Dataset durée réelle
│   │   └── speed.py                ← Normalisation vitesse
│   ├── evaluation/                 ← Évaluation
│   │   ├── __init__.py
│   │   ├── content_metrics.py      ← Métriques de contenu
│   │   └── transition_multilabel_profiles.py ← Profils transition
│   ├── features/                   ← Extraction de features
│   │   ├── mfcc.py                 ← MFCC (caractéristiques acoustiques)
│   │   ├── routing.py              ← Routage basé sur les règles
│   │   └── ssl.py                  ← Features self-supervised (wav2vec)
│   ├── models/                     ← Définitions des modèles
│   │   ├── burst/qalqalah_cnn.py   ← Module Qalqalah (CNN)
│   │   ├── common/                 ← Composants partagés
│   │   │   ├── bilstm_encoder.py   ← Encodeur BiLSTM
│   │   │   ├── ctc_head.py         ← Tête CTC
│   │   │   ├── decoding.py         ← Décodage
│   │   │   ├── losses.py           ← Fonctions de perte
│   │   │   └── rule_head.py        ← Tête classification règles
│   │   ├── content/                ← Modèles de contenu
│   │   │   ├── aligner.py          ← Aligneur contenu
│   │   │   ├── wav2vec_ctc.py      ← wav2vec + CTC
│   │   │   ├── whisper_adapter.py  ← Adaptateur Whisper
│   │   │   └── whisper_ctc.py      ← Whisper CTC
│   │   ├── duration/madd_ghunnah_module.py ← Module Madd/Ghunnah
│   │   ├── fusion/                 ← Fusion et feedback
│   │   │   ├── aggregator.py       ← Agrégation des scores
│   │   │   ├── duration_fusion_calibrator.py ← Calibrateur fusion durée
│   │   │   ├── feedback.py         ← Génération de feedback
│   │   │   └── schemas.py          ← Schémas de données
│   │   ├── routing/learned_router.py ← Routeur appris
│   │   └── transition/             ← Modèles de transition
│   │       ├── idgham_ikhfa_module.py ← Module Ikhfa/Idgham
│   │       └── multilabel_transition_module.py ← Transition multi-label
│   ├── inference/                  ← Pipeline d'inférence
│   │   ├── learned_routing.py      ← Routage appris
│   │   ├── pipeline.py             ← Pipeline principale
│   │   └── transition_multilabel.py ← Transition multi-label
│   ├── training/                   ← Entraînement
│   │   ├── callbacks.py            ← Checkpoints
│   │   ├── engine.py               ← Boucle d'entraînement
│   │   └── metrics.py              ← Métriques
│   ├── scoring/                    ← Scoring
│   │   ├── __init__.py
│   │   ├── error_types.py          ← Types d'erreurs
│   │   ├── inference_adapter.py    ← Adaptateur inférence
│   │   └── weighted_score.py       ← Score pondéré
│   ├── text/                       ← Traitement de texte
│   │   ├── __init__.py
│   │   └── normalization.py        ← Normalisation texte arabe
│   └── utils/                      ← Utilitaires
│       ├── io.py                    ← Entrées/sorties
│       ├── logging.py               ← Logging
│       └── seed.py                  ← Gestion des seeds
│
├── scripts/                        ← Scripts d'exécution (130+ fichiers)
│   ├── README.md                   ← Documentation des scripts
│   ├── burst/                      ← Module Qalqalah (2 scripts)
│   ├── content/                    ← Module contenu (35 scripts)
│   ├── data/                       ← Préparation données (12 scripts)
│   ├── duration/                   ← Module durée (30 scripts)
│   ├── eval/                       ← Évaluation comparative (5 scripts)
│   ├── plots/                      ← Génération de figures (4 scripts)
│   ├── routing/                    ← Routage appris (10 scripts)
│   ├── system/                     ← Système complet (18 scripts)
│   └── transition/                 ← Module transition (18 scripts)
│
├── tests/                          ← Tests unitaires (14 fichiers)
│   ├── test_aggregator.py
│   ├── test_alignment.py
│   ├── test_ayah_content_strict_acceptance.py
│   ├── test_content_metrics.py
│   ├── test_dataset.py
│   ├── test_inference_scoring_adapter.py
│   ├── test_learned_router.py
│   ├── test_models.py
│   ├── test_multilabel_transition.py
│   ├── test_speed.py
│   ├── test_text_normalization.py
│   ├── test_transition_multilabel_inference.py
│   ├── test_weighted_score.py
│   ├── test_whisper_adapter.py
│   └── test_whisper_ctc_model.py
│
└── tajweed-app-frontend/           ← Frontend Next.js
    ├── package.json / package-lock.json
    ├── tsconfig.json / next.config.ts
    ├── postcss.config.mjs / eslint.config.mjs
    ├── public/                     ← Assets statiques
    └── src/
        ├── app/                    ← Pages Next.js
        │   ├── page.tsx            ← Page principale
        │   ├── layout.tsx          ← Layout racine
        │   ├── globals.css         ← Styles globaux
        │   └── favicon.ico
        └── components/             ← Composants React
            ├── ContentFeedback.tsx  ← Feedback contenu
            ├── MushafPreviewCard.tsx ← Aperçu Mushaf
            ├── ReadableFeedback.tsx ← Feedback lisible
            └── SupportedRules.tsx   ← Règles supportées
```

---

## 2. Configuration obligatoire

À faire dans l'ordre **avant toute exécution**.

### Étape 1 — Installer les dépendances

```powershell
# Créer l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate

# Dépendances principales
pip install -r requirements.txt

# Dépendances Whisper (optionnel, pour le content gate)
pip install -r requirements-whisper.txt
```

**Dépendances principales** (`requirements.txt`) : `torch`, `torchaudio`, `transformers`, `librosa`, `fastapi`, `uvicorn`, `pydantic`

**Dépendances Whisper** (`requirements-whisper.txt`) : packages supplémentaires pour le fine-tuning Whisper

### Étape 2 — Placer les checkpoints

Le dossier `checkpoints/` est **ignoré par Git** (`.gitignore`). Les poids des modèles doivent y être placés manuellement.

**Minimum requis pour l'inférence :**

| Fichier | Module | Rôle |
|---|---|---|
| `duration_module.pt` | Durée | Classifie Madd / Ghunnah |
| `transition_module.pt` | Transition | Classifie Ikhfa / Idgham |
| `burst_module.pt` | Burst | Détecte Qalqalah |
| `content_chunked_module_hd96_reciter.pt` | Contenu | Reconnaissance du texte |
| `duration_fusion_calibrator_approved.pt` | Fusion | Combine les scores durée |
| `content_asr_whisper_medium_quran_v2_weighted/` | Content Gate | Vérification contenu (Whisper) |
| `learned_router_v5_retasy_hf_rule_aware_group_text.pt` | Routage | Routeur appris |

### Étape 3 — Vérifier les chemins dans `settings.py`

Éditer `src/tajweed_assessment/settings.py`. Ce fichier définit les chemins globaux utilisés par **tous** les modules :

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ProjectPaths:
    root: Path  # ← Racine du projet

    @property
    def manifests(self) -> Path:
        return self.root / "data" / "manifests"   # ← Les .jsonl

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"           # ← Les .pt et Whisper
```

Par défaut, `root` pointe sur le dossier courant. Adaptez si nécessaire.

### Étape 4 — Vérifier la config du backend

Dans `app/main.py` :
- **Port par défaut** : `8000`
- **Host par défaut** : `0.0.0.0`

Si vous changez le port, mettez à jour l'URL dans le frontend (`tajweed-app-frontend/src/app/page.tsx` et les composants dans `src/components/`).

---

## 3. Fichiers de configuration

### Configs d'architecture modèle (YAML) — Modifiables pour ré-entraîner

| Fichier | Paramètres clés | Quand modifier |
|---|---|---|
| `data.yaml` | `sample_rate: 16000`, `n_mfcc: 40` | Changer fréquence ou features audio |
| `train.yaml` | `seed: 42`, `batch_size: 16`, `lr: 0.001`, `epochs: 50` | Ajuster l'entraînement |
| `model_duration.yaml` | `hidden_size: 128`, `num_layers: 2` | Architecture durée |
| `model_transition.yaml` | `hidden_size: 128`, `num_layers: 2` | Architecture transition |
| `model_burst.yaml` | `kernel_size`, `stride`, `channels` (CNN) | Architecture burst |
| `model_content_hd96.yaml` | `hidden_size: 96`, wav2vec+CTC | **Modèle contenu principal** |
| `model_content_whisper.yaml` | Configuration Whisper | Modèle Whisper contenu |
| `error_weights.yaml` | `duration: 1.0`, `transition: 1.0`, `burst: 1.0`, `content: 2.0` | Ajuster importance relative des erreurs |

### Configs de seuils — Modifiables pour ajuster la sensibilité

| Fichier | Rôle |
|---|---|
| `learned_router_v5_thresholds.yaml` | **Routeur principal** — seuils de confiance |
| `transition_multilabel_thresholds.yaml` | Seuils par classe (ikhfa, idgham...) |

### Configs décodeurs/features (JSON) — Normalement ne pas modifier

| Fichier | Rôle |
|---|---|
| `production_content_gate.json` | Config du content gate de production |
| `whole_system_baseline_v2.json` | Baseline système complet v2 |
| `rule_manifest_json.json` | Règles de Tajweed supportées |
| `content_ayah_decoder_bp12.json` | Décodeur CTC par ayah |
| `content_chunked_decoder_beam_bp04.json` | Décodeur beam search |
| `content_chunked_decoder_eval_lexicon_bp04.json` | Décodeur avec lexique |

---

## 4. Installation

```powershell
# Créer l'environnement
python -m venv .venv
.venv\Scripts\activate

# Installer
pip install -r requirements.txt
pip install -r requirements-whisper.txt   # optionnel
```

**Frontend :**
```powershell
cd tajweed-app-frontend
npm install
```

---

## 5. Lancement

### Backend

```powershell
.venv\Scripts\activate
uvicorn app.main:app --reload
```

Accessible sur **http://127.0.0.1:8000**

### Frontend

```powershell
cd tajweed-app-frontend
npm run dev
```

Accessible sur **http://localhost:3000**

### Utilisation

1. Ouvrir **http://localhost:3000**
2. Choisir un mode de récitation
3. Enregistrer/uploader un audio
4. Le système vérifie le contenu (Content Gate)
5. Si accepté, les modules Tajweed analysent l'audio
6. Diagnostic et feedback affichés

---

## 6. Modules principaux

### Content Gate — Vérification du contenu
- Whisper Medium fine-tuné sur le Quran
- Normalisation des muqattaat
- **73.96%** exact match, **98.17%** char accuracy

### Module Durée — Madd, Ghunnah
- BiLSTM sur features MFCC + localisation temporelle
- **99.27%** accuracy

### Module Transition — Ikhfa, Idgham
- Support multi-label (plusieurs règles par clip)
- **91.01%** accuracy

### Module Burst — Qalqalah
- CNN sur features MFCC
- **87.54%** accuracy (seuil 0.47)

### Feedback
- Transforme les sorties techniques en messages compréhensibles

---

## 7. Scripts

### Entraînement

```powershell
python scripts/duration/train_duration.py
python scripts/transition/train_transition.py
python scripts/burst/train_burst.py
python scripts/content/train_chunked_content.py
```

### Évaluation

```powershell
python scripts/system/evaluate_modular_suite.py
```

### Inférence

```powershell
python scripts/system/run_inference.py --manifest data/manifests/retasy_transition_subset.jsonl --sample-index 1
```

### Tests

```powershell
python -m pytest tests -q
```

---

## 8. Données

### `data/manifests/` — 87 fichiers .jsonl

| Fichier | Contenu |
|---|---|
| `retasy_train.jsonl` | Dataset principal |
| `retasy_duration_alignment_corpus_torchaudio_strict.jsonl` | Alignement durée |
| `retasy_transition_subset.jsonl` | Sous-ensemble transition |
| `retasy_burst_subset.jsonl` | Sous-ensemble burst |
| `retasy_content_chunks.jsonl` | Contenu par chunks |
| `quran_content_reference_full.jsonl` | Référence texte Quran |
| `quran_tajweed_reference_full.jsonl` | Référence règles Tajweed |

### `data/analysis/` — Résultats d'évaluation

- `thesis_ablation_v2/` — Résumés d'ablation thèse
- `modular_suite_*.json` — Évaluations modulaires
- `whole_system_status_report_v2.md` — Rapport final

### `data/raw/` — Audio brut (ignoré par Git)

Fichiers WAV/MP3 d'entraînement (HuggingFace Quran-MD, Retasy).

---

## 9. Checkpoints

Le dossier `checkpoints/` (~70 Go) contient tous les poids des modèles entraînés :
- **`.pt`** : poids PyTorch (quelques Mo à ~380 Mo)
- **Dossiers Whisper** : `model.safetensors`, `config.json`, `tokenizer.json`...

Les 5 dossiers Whisper : `content_asr_whisper_small_quran_v1_*` (3.7-6.4 Go), `content_asr_whisper_medium_quran_v1_*` (12 Go), `content_asr_whisper_medium_quran_v2_weighted/` (23 Go).

---

## 10. Résultats clés

| Composant | Métrique | Résultat |
|---|---|---|
| Content gate | Exact après normalisation | 73.96% |
| Content gate | Char accuracy après normalisation | 98.17% |
| Module durée | Accuracy | 99.27% |
| Module transition | Accuracy | 91.01% |
| Module burst | Accuracy | 87.54% |

---

## 11. Remarques

1. **Checkpoints** : `checkpoints/` est ignoré par Git. Les `.pt` et dossiers Whisper doivent être récupérés séparément.
2. **Audio brut** : `data/raw/` est ignoré par Git. Contient les WAV/MP3 d'entraînement.
3. **Données intermédiaires** : `data/interim/`, `data/processed/`, `data/uploads/` sont ignorés par Git.
4. **Fichiers de config** : Les YAML/JSON dans `configs/` sont versionnés et nécessaires au fonctionnement.
5. **settings.py** : Vérifier `ProjectPaths.root` avant toute exécution.

---

*Dernière mise à jour : 21 juin 2026 — PFE Tajweed Modular Assessment*