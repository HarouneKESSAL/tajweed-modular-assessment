from dataclasses import dataclass

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


@dataclass
class WhisperTranscription:
    text: str


class WhisperContentAdapter:
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        language: str = "arabic",
        task: str = "transcribe",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.processor = WhisperProcessor.from_pretrained(
            model_name,
            language=language,
            task=task,
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language,
            task=task,
        )
        self.model.config.forced_decoder_ids = forced_decoder_ids

    @torch.no_grad()
    def transcribe_array(self, audio, sampling_rate: int) -> WhisperTranscription:
        inputs = self.processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)
        predicted_ids = self.model.generate(input_features)
        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]
        return WhisperTranscription(text=text.strip())