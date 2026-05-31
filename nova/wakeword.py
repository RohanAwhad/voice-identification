from __future__ import annotations

import numpy as np
import pyaudio
from openwakeword.model import Model

AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHUNK = 1280  # 80ms at 16kHz

WAKE_THRESHOLD = 0.5
WAKE_MODEL = "hey_mycroft"
WARMUP_FRAMES = 5  # openWakeWord returns 0.0 for first 5 frames (~400ms)


class WakeWordDetector:
    def __init__(self) -> None:
        self.model = Model(
            wakeword_models=[WAKE_MODEL],
            vad_threshold=0.0,
            inference_framework="tflite",
            enable_speex_noise_suppression=False,
        )

        self.threshold = WAKE_THRESHOLD
        self.pya = pyaudio.PyAudio()
        self.stream: pyaudio.Stream | None = None

    def start(self) -> None:
        self.model.reset()
        self.stream = self.pya.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK,
        )

    def stop(self) -> None:
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def wait_for_wake(self) -> None:
        for _ in range(WARMUP_FRAMES):
            pcm = np.frombuffer(
                self.stream.read(AUDIO_CHUNK, exception_on_overflow=False),
                dtype=np.int16,
            )
            self.model.predict(pcm)

        while True:
            pcm = np.frombuffer(
                self.stream.read(AUDIO_CHUNK, exception_on_overflow=False),
                dtype=np.int16,
            )

            prediction = self.model.predict(pcm)
            for name, score in prediction.items():
                if score > self.threshold:
                    self.model.reset()
                    return

    def close(self) -> None:
        self.stop()
        self.pya.terminate()
