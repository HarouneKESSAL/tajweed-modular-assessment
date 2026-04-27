from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import torch
from tajweed_assessment.data.manifests import load_manifest, save_manifest
from tajweed_assessment.features.mfcc import extract_mfcc_features
from tajweed_assessment.settings import ProjectPaths

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/train_manifest.json")
    args = parser.parse_args()

    paths = ProjectPaths(PROJECT_ROOT)
    manifest_path = PROJECT_ROOT / args.manifest
    entries = load_manifest(manifest_path)
    for entry in entries:
        feat = extract_mfcc_features(entry.audio_path)
        out_path = paths.processed / f"{entry.sample_id}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feat, out_path)
        entry.feature_path = str(out_path)
    save_manifest(entries, manifest_path)
    print(f"updated {manifest_path}")

if __name__ == "__main__":
    main()

