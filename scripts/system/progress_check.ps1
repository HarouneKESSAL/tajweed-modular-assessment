param(
    [string]$DurationManifest = "data\manifests\retasy_duration_alignment_corpus_torchaudio_strict.jsonl",
    [string]$DurationCheckpoint = "checkpoints\duration_module.pt",
    [string]$RealDurationManifest = "data\manifests\retasy_duration_subset.jsonl",
    [string]$RealDurationCheckpoint = "checkpoints\real_duration_classifier.pt",
    [string]$LocalizedManifest = "data\alignment\duration_time_projection_strict.jsonl",
    [string]$LocalizedCheckpoint = "checkpoints\localized_duration_model.pt",
    [switch]$RunTrainDuration,
    [switch]$RunEvaluateDuration,
    [switch]$RunPredictRealDuration,
    [switch]$RunPredictLocalized,
    [switch]$RunTests
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 16 + " " + $Title + " " + "=" * 16)
}

function Invoke-Step {
    param(
        [string]$Label,
        [scriptblock]$Action
    )

    Write-Section $Label
    & $Action
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtualenv Python not found at $python"
}

Set-Location $repoRoot

Invoke-Step "Git Status" { git status --short }
Invoke-Step "Git Diff Stat" { git diff --stat }
Invoke-Step "Data Config" { Get-Content "configs\data.yaml" }
Invoke-Step "Duration Model Config" { Get-Content "configs\model_duration.yaml" }
Invoke-Step "Train Config" { Get-Content "configs\train.yaml" }

if ($RunTrainDuration) {
    Invoke-Step "Train Duration" {
        & $python "scripts\train_duration.py" "--manifest" $DurationManifest
    }
}

if ($RunEvaluateDuration) {
    Invoke-Step "Evaluate Duration" {
        & $python "scripts\evaluate.py" "--checkpoint" $DurationCheckpoint "--manifest" $DurationManifest "--split" "val"
    }
}

if ($RunPredictRealDuration) {
    Invoke-Step "Predict Real Duration" {
        & $python "scripts\predict_real_duration_classifier.py" "--manifest" $RealDurationManifest "--checkpoint" $RealDurationCheckpoint "--normalize-speed"
    }
}

if ($RunPredictLocalized) {
    Invoke-Step "Predict Localized Duration" {
        & $python "scripts\predict_localized_duration.py" "--manifest" $LocalizedManifest "--checkpoint" $LocalizedCheckpoint "--normalize-speed"
    }
}

if ($RunTests) {
    Invoke-Step "Tests" {
        & $python "-m" "pytest" "tests\test_models.py" "tests\test_dataset.py" "tests\test_speed.py" "-q"
    }
}
