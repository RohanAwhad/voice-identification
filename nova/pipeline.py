from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

from nova.audio import UtteranceCapture
from nova.llm import NovaLLM
from nova.speaker import load_voiceprint, default_voiceprint_path
from nova.stt import Transcriber
from nova.tts import NovaTTS
from nova.wakeword import WakeWordDetector


class NovaPipeline:
    def __init__(self) -> None:
        self.voiceprint_path = default_voiceprint_path()
        self.voiceprint = load_voiceprint(self.voiceprint_path)

        print("Loading models...", flush=True)

        print("  SpeakerID (WeSpeaker)...", end=" ", flush=True)
        from nova.speaker import SpeakerVerifier

        self.verifier = SpeakerVerifier()
        print("done")

        print("  STT (faster-whisper turbo)...", end=" ", flush=True)
        self.transcriber = Transcriber()
        print("done")

        print("  LLM (DeepSeek v4-flash)...", end=" ", flush=True)
        self.llm = NovaLLM()
        print("done")

        print("  TTS (Kokoro)...", end=" ", flush=True)
        self.tts = NovaTTS()
        print("done")

        print("  Wake word (openWakeWord)...", end=" ", flush=True)
        self.wakeword = WakeWordDetector()
        print("done")

        print("  VAD (Silero)...", end=" ", flush=True)
        self.capture = UtteranceCapture()
        print("done")

        print("Ready.", flush=True)

    def run(self) -> None:
        self.wakeword.start()
        print("Listening for wake word 'hey mycroft'...\n", flush=True)

        while True:
            self.wakeword.wait_for_wake()
            print("Wake word detected!", flush=True)
            self.wakeword.stop()

            self.capture.start()
            print("Listening... (Ctrl+C when done)", flush=True)
            try:
                utterance = self.capture.capture()
            except KeyboardInterrupt:
                print("", flush=True)
                utterance = self.capture.drain()
                if utterance is None or len(utterance) < 16000 * 0.5:
                    print("No speech captured.", flush=True)
                    self.capture.stop()
                    self.wakeword.start()
                    print("\nListening for wake word 'hey mycroft'...\n", flush=True)
                    continue
            self.capture.stop()
            print(f"Captured {len(utterance) / 16000:.1f}s of speech.", flush=True)

            with ThreadPoolExecutor(max_workers=2) as executor:
                t0 = time.time()
                speaker_future = executor.submit(
                    self.verifier.verify, utterance, self.voiceprint
                )
                transcript_future = executor.submit(self.transcriber.transcribe, utterance)

                (identity, confidence) = speaker_future.result()
                transcript = transcript_future.result()
                t1 = time.time()

            print(f"Speaker: {identity.value} ({confidence:.3f}) [{t1 - t0:.1f}s]", flush=True)
            print(f'Transcript: "{transcript}"', flush=True)

            speaker_label = identity.value
            response = self.llm.ask(transcript, speaker_label)
            t2 = time.time()

            print(f"Nova: {response} [{t2 - t1:.1f}s]", flush=True)
            self.tts.speak(response)
            t3 = time.time()
            print(f"TTS+playback [{t3 - t2:.1f}s]", flush=True)

            self.wakeword.start()
            print("\nListening for wake word 'hey mycroft'...\n", flush=True)
