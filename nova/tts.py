from __future__ import annotations

import os
import subprocess
import tempfile

import soundfile as sf
from kokoro_tts import Kokoro


DEFAULT_VOICE = "af_bella"


class NovaTTS:
    def __init__(self, voice: str = DEFAULT_VOICE) -> None:
        self.kk = Kokoro()
        self.voice = voice

    def speak(self, text: str) -> None:
        audio = self.kk.generate(text, self.voice)

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, audio, self.kk.SAMPLING_RATE)
        subprocess.run(["afplay", path], check=True)
        os.unlink(path)
