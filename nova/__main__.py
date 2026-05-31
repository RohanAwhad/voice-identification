from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyaudio

from nova.speaker import (
    ENROLL_SENTENCES,
    N_SAMPLES,
    SAMPLE_LENGTH_S,
    SpeakerVerifier,
    default_voiceprint_path,
    save_voiceprint,
)

AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHUNK = 1024


def _record_seconds(seconds: int) -> np.ndarray:
    pya = pyaudio.PyAudio()
    stream = pya.open(
        format=AUDIO_FORMAT,
        channels=AUDIO_CHANNELS,
        rate=AUDIO_RATE,
        input=True,
        frames_per_buffer=AUDIO_CHUNK,
    )

    frames: list[bytes] = []
    n_chunks = int(AUDIO_RATE / AUDIO_CHUNK * seconds)
    for _ in range(n_chunks):
        frames.append(stream.read(AUDIO_CHUNK, exception_on_overflow=False))

    stream.stop_stream()
    stream.close()
    pya.terminate()

    all_pcm = b"".join(frames)
    return np.frombuffer(all_pcm, dtype=np.int16)


def enroll() -> None:
    verifier = SpeakerVerifier()

    print(f"=== Nova Enrollment ===\n")
    print(f"Will record {N_SAMPLES} samples, {SAMPLE_LENGTH_S}s each.")
    print(f"Speak clearly into the microphone.\n")

    samples: list[np.ndarray] = []
    for i in range(N_SAMPLES):
        sentence = ENROLL_SENTENCES[i % len(ENROLL_SENTENCES)]
        print(f"Recording {i+1}/{N_SAMPLES}")
        print(f'  Please read: "{sentence}"')
        input("  Press Enter to start recording...")

        audio = _record_seconds(SAMPLE_LENGTH_S)
        samples.append(audio)

        print(f"  Saved ({audio.shape[0] / AUDIO_RATE:.1f}s)\n")

    print("Extracting voiceprint...")
    voiceprint = verifier.enroll(samples)

    path = default_voiceprint_path()
    save_voiceprint(voiceprint, path)

    print(f"Enrollment complete. Voiceprint saved to: {path}")


def run_assistant() -> None:
    from nova.pipeline import NovaPipeline

    voiceprint_path = default_voiceprint_path()
    if not voiceprint_path.exists():
        print(f"No voiceprint found at {voiceprint_path}")
        print("Run 'python nova.py enroll' first.")
        sys.exit(1)

    print("Starting Nova...")
    print(f"  Voiceprint: {voiceprint_path}")
    print(f"  Wake word: hey mycroft")
    print(f"  Say 'hey mycroft' to start, then speak your request.\n")

    pipeline = NovaPipeline()
    pipeline.run()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "enroll":
        enroll()
    else:
        run_assistant()
