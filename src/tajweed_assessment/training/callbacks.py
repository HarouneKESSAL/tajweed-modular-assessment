from dataclasses import dataclass
from pathlib import Path
from tajweed_assessment.utils.io import save_checkpoint

@dataclass
class ModelCheckpoint:
    directory: Path
    filename: str = "model.pt"

    def __post_init__(self) -> None:
        self.best_value = float("inf")

    def step(self, value: float, state: dict) -> bool:
        if value < self.best_value:
            self.best_value = value
            save_checkpoint(state, self.directory / self.filename)
            return True
        return False
