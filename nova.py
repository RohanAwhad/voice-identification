from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyaudio

from nova.speaker import (
    ENROLL_SENTENCES,
    N_SAMPLES,
    SAMPLE_LENGTH_S,
    SpeakerVerifier,
    default_voiceprint_path,
    load_voiceprint,
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


def cmd_enroll(args: argparse.Namespace) -> None:
    verifier = SpeakerVerifier()

    print("=== Nova Enrollment ===")
    print(f"Will record {N_SAMPLES} samples, {SAMPLE_LENGTH_S}s each.", flush=True)
    print("Speak clearly into the microphone.\n", flush=True)

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


def cmd_verify(args: argparse.Namespace) -> None:
    from nova.speaker import SpeakerIdentity

    voiceprint_path = default_voiceprint_path()
    if not voiceprint_path.exists():
        print(f"No voiceprint found at {voiceprint_path}")
        print("Run 'python nova.py enroll' first.")
        sys.exit(1)

    voiceprint = load_voiceprint(voiceprint_path)
    verifier = SpeakerVerifier()

    duration = getattr(args, "duration", 5)
    print("=== Nova Identify ===")
    print(f"Loaded voiceprint from: {voiceprint_path}")
    print(f"Press Enter to start recording {duration} seconds of speech...")
    input()

    print("Recording...")
    audio = _record_seconds(duration)
    print(f"Captured {audio.shape[0] / AUDIO_RATE:.1f}s of audio.")

    identity, confidence = verifier.verify(audio, voiceprint)
    print(f"\nResult: {identity.value} (confidence: {confidence:.4f})")
    if identity == SpeakerIdentity.ROHAN:
        print("You are recognized as Rohan.")
    else:
        print("You are NOT recognized as Rohan.")


def cmd_run(args: argparse.Namespace) -> None:
    from nova.pipeline import NovaPipeline

    voiceprint_path = default_voiceprint_path()
    if not voiceprint_path.exists():
        print(f"No voiceprint found at {voiceprint_path}")
        print("Run 'python nova.py enroll' first.")
        sys.exit(1)

    print("Starting Nova...")
    print(f"  Voiceprint: {voiceprint_path}")
    print(f"  Wake word: hey mycroft")
    print("  Say 'hey mycroft' to start, then speak your request.\n", flush=True)

    pipeline = NovaPipeline()
    pipeline.run()


def main() -> None:
    parser = argparse.ArgumentParser(prog="nova", description="Sassy voice assistant with speaker identification")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("enroll", help="Record 5 voice samples and create voiceprint")
    sub.add_parser("run", help="Start the always-listening assistant loop")

    verify_parser = sub.add_parser("verify", help="Record speech and check against saved voiceprint")
    verify_parser.add_argument("--duration", type=int, default=5, help="Recording duration in seconds (default: 5)")

    args = parser.parse_args()

    handlers = {
        "enroll": cmd_enroll,
        "verify": cmd_verify,
        "run": cmd_run,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
