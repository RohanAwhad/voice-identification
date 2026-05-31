from __future__ import annotations

import re

import sounddevice as sd
from kokoro_tts import Kokoro

DEFAULT_VOICE = "af_bella"


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class NovaTTS:
    def __init__(self, voice: str = DEFAULT_VOICE) -> None:
        self.kk = Kokoro()
        self.voice = voice
        self.rate = self.kk.SAMPLING_RATE

    def speak(self, text: str) -> None:
        sentences = _split_sentences(text)
        for sentence in sentences:
            audio = self.kk.generate(sentence, self.voice)
            sd.play(audio, self.rate)
            sd.wait()
