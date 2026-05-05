from dataclasses import dataclass


@dataclass(frozen=True)
class TajweedError:
    module: str
    error_type: str
    confidence: float = 1.0
    location: str | None = None
    expected: str | None = None
    predicted: str | None = None
    message: str | None = None