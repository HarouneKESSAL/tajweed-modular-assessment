from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.data.dataset import ToyDurationDataset
from tajweed_assessment.inference.pipeline import TajweedInferencePipeline
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.utils.seed import seed_everything

def main() -> None:
    seed_everything(7)
    sample = ToyDurationDataset(n_samples=1, seed=7)[0]
    model = DurationRuleModule()
    pipeline = TajweedInferencePipeline(duration_module=model)
    result = pipeline.run_duration_only(
        x=sample["x"],
        input_length=len(sample["x"]),
        canonical_phonemes=sample["phoneme_targets"].tolist(),
        canonical_rules=sample["canonical_rules"].tolist(),
        word=sample["word"],
    )
    print(result["report"])
    print(result["feedback"])

if __name__ == "__main__":
    main()
