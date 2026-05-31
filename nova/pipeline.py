from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from nova.audio import UtteranceCapture
from nova.llm import NovaLLM
from nova.speaker import SpeakerIdentity, SpeakerVerifier, load_voiceprint, default_voiceprint_path
from nova.stt import Transcriber
from nova.tts import NovaTTS
from nova.wakeword import WakeWordDetector


class NovaPipeline:
    def __init__(self) -> None:
        self.voiceprint_path = default_voiceprint_path()
        self.voiceprint = load_voiceprint(self.voiceprint_path)

        self.verifier = SpeakerVerifier()
        self.transcriber = Transcriber()
        self.llm = NovaLLM()
        self.tts = NovaTTS()

        self.wakeword = WakeWordDetector()
        self.capture = UtteranceCapture()

    def run(self) -> None:
        self.wakeword.start()

        while True:
            self.wakeword.wait_for_wake()

            self.wakeword.stop()

            self.capture.start()
            utterance = self.capture.capture()
            self.capture.stop()

            with ThreadPoolExecutor(max_workers=2) as executor:
                speaker_future = executor.submit(
                    self.verifier.verify, utterance, self.voiceprint
                )
                transcript_future = executor.submit(self.transcriber.transcribe, utterance)

                (identity, confidence) = speaker_future.result()
                transcript = transcript_future.result()

            speaker_label = identity.value
            response = self.llm.ask(transcript, speaker_label)

            self.tts.speak(response)

            self.wakeword.start()
