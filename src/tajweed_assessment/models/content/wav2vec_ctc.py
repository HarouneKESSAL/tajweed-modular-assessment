import torch
import torch.nn as nn
from tajweed_assessment.features.ssl import Wav2VecFeatureExtractor
from tajweed_assessment.models.common.bilstm_encoder import BiLSTMEncoder
from tajweed_assessment.models.common.ctc_head import CTCHead

class ContentVerificationModule(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_phonemes: int = 11) -> None:
        super().__init__()
        self.ssl = Wav2VecFeatureExtractor()
        feature_dim = getattr(self.ssl, "output_dim", 64)
        self.encoder = BiLSTMEncoder(input_dim=feature_dim, hidden_dim=hidden_dim, num_layers=1, dropout=0.1)
        self.ctc_head = CTCHead(self.encoder.output_dim, num_phonemes)

    def forward_features(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x, lengths)
        return self.ctc_head.log_probs(encoded)

    def forward_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        feats = self.ssl(waveform).unsqueeze(0)
        lengths = torch.tensor([feats.size(1)], dtype=torch.long)
        return self.forward_features(feats, lengths)
