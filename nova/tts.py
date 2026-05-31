from __future__ import annotations

import queue
import re
import threading

import numpy as np
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
        if not sentences:
            return

        audio_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=2)

        def generate() -> None:
            for s in sentences:
                audio_q.put(self.kk.generate(s, self.voice))
            audio_q.put(None)

        t = threading.Thread(target=generate, daemon=True)
        t.start()

        while True:
            audio = audio_q.get()
            if audio is None:
                break
            sd.play(audio, self.rate)
            sd.wait()

        t.join()
