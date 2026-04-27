from dataclasses import dataclass
from typing import Literal
import torch
from tajweed_assessment.data.labels import id_to_rule
from tajweed_assessment.features.mfcc import extract_mfcc_features
from tajweed_assessment.features.ssl import DummySSLFeatureExtractor, Wav2VecFeatureExtractor

def build_feature_view(audio_path: str, view: Literal["mfcc", "ssl", "hybrid"] = "mfcc") -> dict[str, torch.Tensor]:
    if view == "mfcc":
        return {"mfcc": extract_mfcc_features(audio_path)}
    if view == "ssl":
        extractor = Wav2VecFeatureExtractor()
        return {"ssl": extractor.forward_path(audio_path)}
    if view == "hybrid":
        mfcc = extract_mfcc_features(audio_path)
        ssl = DummySSLFeatureExtractor(output_dim=64).from_mfcc(mfcc)
        return {"mfcc": mfcc, "ssl": ssl}
    raise ValueError(f"Unknown feature view: {view}")


@dataclass(frozen=True)
class RoutingPlan:
    use_duration: bool
    use_transition: bool
    use_burst: bool


def build_routing_plan(canonical_rules: list[int] | list[str]) -> RoutingPlan:
    normalized: list[str] = []
    for rule in canonical_rules:
        if isinstance(rule, int):
            normalized.append(id_to_rule.get(rule, "none"))
        else:
            normalized.append(str(rule))

    use_transition = any(rule in {"ikhfa", "idgham"} for rule in normalized)
    use_burst = any(rule == "qalqalah" for rule in normalized)
    use_duration = any(rule in {"madd", "ghunnah"} for rule in normalized) or not (use_transition or use_burst)

    return RoutingPlan(
        use_duration=use_duration,
        use_transition=use_transition,
        use_burst=use_burst,
    )
