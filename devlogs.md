# Nova Devlog

## 2026-05-31 — v1 Implementation

### Completed
- All 7 PRD docs written in `.dingllm/prds/v1/`
- Full project structure: `nova/` package with 6 modules + CLI entry
- `nova/speaker.py` — WeSpeaker wrapper (enroll, verify, voiceprint I/O)
- `nova/stt.py` — faster-whisper `turbo` model, int8 compute
- `nova/llm.py` — DeepSeek v4-flash with thinking disabled
- `nova/tts.py` — Kokoro TTS + `afplay` playback
- `nova/audio.py` — Silero VADIterator utterance capture with ring buffer
- `nova/wakeword.py` — openWakeWord "hey mycroft" detection
- `nova/pipeline.py` — Main loop: wake → capture → speaker ID + STT (parallel) → LLM → TTS
- `nova/__main__.py` — CLI: `python nova.py enroll` and `python nova.py`
- All 7 smoke tests pass (venv, model loading, API surface)

### Dependency issues resolved
- Python 3.14 → 3.12 (onnxruntime wheel availability)
- s3prl patched: `torchaudio.set_audio_backend("sox_io")` removed (absent in torchaudio 2.0+)
- s3prl patched: `torchaudio.sox_effects.apply_effects_tensor` import removed
- torchcodec installed (needed by torchaudio 2.11 `load()`)
- kokoro voice download fixed: changed commit hash to `main` branch (Rohan's fork had stale hashes)

### Source files
```
nova/
├── __init__.py
├── __main__.py      # CLI entry: enroll / run
├── speaker.py        # WeSpeaker wrapper
├── stt.py            # faster-whisper
├── llm.py            # DeepSeek API
├── tts.py            # Kokoro TTS + playback
├── audio.py          # Mic capture + Silero VAD
├── wakeword.py       # openWakeWord
└── pipeline.py       # Main loop
pyproject.toml
play.py               # Smoke test suite
```

### Remaining
- Full end-to-end test with real mic + DeepSeek API key
- Wake word threshold calibration
- Speaker ID threshold calibration on Rohan + other voices
- Custom "Nova" wake word model training
- Logging infrastructure (loguru)
