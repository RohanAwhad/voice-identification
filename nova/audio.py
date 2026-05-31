from __future__ import annotations

from collections import deque

import numpy as np
import pyaudio
import torch
from silero_vad import VADIterator, load_silero_vad

AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHUNK = 512  # 32ms at 16kHz — required by Silero VADIterator

VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 1500
VAD_MIN_SPEECH_MS = 250
VAD_SPEECH_PAD_MS = 30
RING_BUFFER_DURATION_S = 30


class UtteranceCapture:
    def __init__(self) -> None:
        self.pya = pyaudio.PyAudio()

        model = load_silero_vad(onnx=True)
        self.silero = model
        self.vad = VADIterator(
            model,
            threshold=VAD_THRESHOLD,
            sampling_rate=AUDIO_RATE,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        )

        self.ring: deque[tuple[int, np.ndarray]] = deque(
            maxlen=int(AUDIO_RATE * RING_BUFFER_DURATION_S / AUDIO_CHUNK)
        )
        self.sample_offset = 0
        self.speech_start_offset: int | None = None
        self.stream: pyaudio.Stream | None = None

    def start(self) -> None:
        self.vad.reset_states()
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

    def capture(self) -> np.ndarray:
        self.sample_offset = 0
        self.speech_start_offset = None
        self.ring.clear()

        while True:
            pcm = np.frombuffer(
                self.stream.read(AUDIO_CHUNK, exception_on_overflow=False),
                dtype=np.int16,
            )

            self.ring.append((self.sample_offset, pcm))
            self.sample_offset += len(pcm)

            audio_tensor = torch.from_numpy(pcm.astype(np.float32))
            event = self.vad(audio_tensor, return_seconds=False)

            if event:
                if "start" in event:
                    self.speech_start_offset = event["start"]
                elif "end" in event and self.speech_start_offset is not None:
                    utterance = np.concatenate(
                        [
                            chunk
                            for off, chunk in self.ring
                            if self.speech_start_offset <= off < event["end"]
                        ]
                    )
                    self.vad.reset_states()
                    return utterance

    def drain(self) -> np.ndarray | None:
        """Return partial utterance captured so far. None if no speech started."""
        if self.speech_start_offset is None:
            return None
        utterance = np.concatenate(
            [
                chunk
                for off, chunk in self.ring
                if self.speech_start_offset <= off < self.sample_offset
            ]
        )
        self.vad.reset_states()
        return utterance

    def close(self) -> None:
        self.stop()
        self.pya.terminate()
