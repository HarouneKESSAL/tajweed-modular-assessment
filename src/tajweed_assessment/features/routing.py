from typing import Literal
import torch
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
