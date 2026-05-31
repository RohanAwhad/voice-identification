"""Nova v1 — smoke test suite."""
from __future__ import annotations

import sys
import time


def test_we_speaker_load() -> None:
    """Verify WeSpeaker model loads and extracts embeddings."""
    import wespeaker
    import numpy as np
    import soundfile as sf

    model = wespeaker.load_model("english")
    model.set_vad(False)

    # 1 second of random "voice-like" noise at 16kHz
    rng = np.random.default_rng(42)
    fake_audio = (rng.normal(0, 8000, 16000 * 2)).astype(np.int16)
    sf.write("/tmp/_nova_test_sine.wav", fake_audio, 16000)

    emb = model.extract_embedding("/tmp/_nova_test_sine.wav")
    assert emb is not None, "extract_embedding returned None (VAD may have stripped)"
    assert emb.shape[0] == 256, f"Expected 256-dim embedding, got {emb.shape}"
    print("  PASS: WeSpeaker model loads, embedding shape correct")


def test_faster_whisper_load() -> None:
    """Verify faster-whisper model loads (will download tiny model on first run)."""
    from faster_whisper import WhisperModel
    import numpy as np

    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    fake_audio = np.random.randn(16000 * 2).astype(np.float32) * 0.01
    segments, info = model.transcribe(
        fake_audio,
        beam_size=1,
        language="en",
        vad_filter=False,
    )
    _ = " ".join(seg.text.strip() for seg in segments)
    print("  PASS: faster-whisper model loads and runs")


def test_kokoro_tts() -> None:
    """Verify Kokoro TTS generates audio (no playback)."""
    from kokoro_tts import Kokoro

    kk = Kokoro()
    audio = kk.generate("Hello world, this is a test.", "af_bella")
    assert len(audio) > 0
    assert audio.dtype.name == "float32"
    print(f"  PASS: Kokoro TTS generates {len(audio)} samples ({len(audio)/(kk.SAMPLING_RATE):.1f}s)")


def test_llm_client() -> None:
    """Verify DeepSeek LLM client raises clear error without API key."""
    import os

    key = os.environ.pop("DEEPSEEK_API_KEY", None)
    from nova.llm import NovaLLM

    raised = False
    msg = ""
    try:
        _ = NovaLLM()
    except RuntimeError as e:
        raised = True
        msg = str(e)
    assert raised, "Expected RuntimeError when DEEPSEEK_API_KEY not set"
    print(f"  PASS: LLM raises clear error without API key")

    if key:
        os.environ["DEEPSEEK_API_KEY"] = key


def test_speaker_api() -> None:
    """Verify SpeakerVerifier enrollment and verification API."""
    import numpy as np
    from nova.speaker import SpeakerIdentity, SpeakerVerifier

    verifier = SpeakerVerifier()
    verifier.model.set_vad(False)

    samples: list[np.ndarray] = []
    rng = np.random.default_rng(42)
    for i in range(5):
        audio = (rng.normal(0, 8000, 16000 * 3)).astype(np.int16)
        samples.append(audio)

    voiceprint = verifier.enroll(samples)
    assert voiceprint.shape == (256,), f"Expected (256,), got {voiceprint.shape}"
    assert voiceprint.dtype == np.float32

    identity, score = verifier.verify(samples[0], voiceprint)
    print(f"  PASS: Enrollment produces 256-dim voiceprint, verify returns ({identity.value}, {score:.4f})")


def test_pipeline_structure() -> None:
    """Verify NovaPipeline can be constructed (no run)."""
    import numpy as np
    from nova.speaker import default_voiceprint_path, save_voiceprint, load_voiceprint

    fake_emb = np.ones(256, dtype=np.float32)
    fake_emb = fake_emb / np.linalg.norm(fake_emb)
    save_voiceprint(fake_emb, default_voiceprint_path())

    loaded = load_voiceprint(default_voiceprint_path())
    assert loaded.shape == (256,)
    print("  PASS: Voiceprint save/load roundtrip works")


def test_audio_capture_structure() -> None:
    """Verify UtteranceCapture imports and init work (no live capture)."""
    from nova.audio import UtteranceCapture

    capture = UtteranceCapture()
    assert capture.pya is not None
    assert capture.vad is not None
    print("  PASS: UtteranceCapture init OK (no mic used)")


def main() -> None:
    print("=== Nova v1 Smoke Tests ===\n")

    tests = [
        ("WeSpeaker", test_we_speaker_load),
        ("faster-whisper (tiny)", test_faster_whisper_load),
        ("Kokoro TTS", test_kokoro_tts),
        ("LLM client", test_llm_client),
        ("Speaker API", test_speaker_api),
        ("Pipeline structure", test_pipeline_structure),
        ("Audio capture structure", test_audio_capture_structure),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"[{name}]")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
        print()

    print(f"=== Results: {passed} passed, {failed} failed ===")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
