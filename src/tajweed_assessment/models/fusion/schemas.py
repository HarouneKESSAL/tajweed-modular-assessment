from dataclasses import asdict, dataclass, field
from typing import Any, List

@dataclass
class DiagnosisError:
    position: int
    type: str
    rule: str | None = None
    detail: str | None = None
    expected: str | None = None
    predicted: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class DiagnosisReport:
    word: str
    canonical_phonemes: List[str]
    predicted_phonemes: List[str]
    errors: List[DiagnosisError]

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "canonical_phonemes": self.canonical_phonemes,
            "predicted_phonemes": self.predicted_phonemes,
            "errors": [asdict(e) for e in self.errors],
        }
