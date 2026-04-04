from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.data.manifests import ManifestEntry, save_manifest
from tajweed_assessment.settings import ProjectPaths

def main() -> None:
    paths = ProjectPaths(PROJECT_ROOT)
    entries = [
        ManifestEntry(
            sample_id="example_001",
            audio_path=str(paths.raw / "example_001.wav"),
            canonical_phonemes=["m", "a", "l", "i", "k"],
            canonical_rules=["none", "madd", "none", "none", "none"],
            text="Malik",
        )
    ]
    save_manifest(entries, paths.manifests / "train_manifest.json")
    print(f"wrote {paths.manifests / 'train_manifest.json'}")

if __name__ == "__main__":
    main()
