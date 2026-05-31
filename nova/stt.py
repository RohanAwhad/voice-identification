from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size: str = "turbo") -> None:
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

    def transcribe(self, audio: np.ndarray) -> str:
        audio_f32 = audio.astype(np.float32) / 32768.0

        segments, _info = self.model.transcribe(
            audio_f32,
            beam_size=1,
            language="en",
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments)
