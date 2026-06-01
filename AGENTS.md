# AGENTS.md — Nova Voice Assistant

## Quick commands
```bash
.venv/bin/python play.py                              # Run all smoke tests
.venv/bin/python nova.py enroll                       # Enroll voiceprint
.venv/bin/python nova.py verify                       # Verify speaker (5s)
.venv/bin/python nova.py verify --duration 10         # Verify speaker (10s)
.venv/bin/python nova.py run                          # Start assistant loop
```

## Environment
- **Python 3.12 only** — `.python-version` says 3.14 but onnxruntime has no wheels for 3.14. Use `python@3.12`.
- **Package manager**: `uv` with `hatchling` build backend.
- **Source dependencies** installed from git via `uv pip install -e . "git+..."` — they are NOT listed as editable local deps. Re-run the full setup command after any `uv sync`.
- **`DEEPSEEK_API_KEY`** env var required for LLM. Must be set before import of `nova.llm`.
- **Voiceprint** stored at `~/.nova/voiceprint.npy`.

## Setup gotchas
- `brew install espeak-ng` required (Kokoro TTS dep).
- After install, patch s3prl (see README lines 15-16) — removes dead `torchaudio.set_audio_backend("sox_io")` and broken import in mos_prediction.
- `torchcodec` must be installed separately (pip install, not in pyproject.toml deps) — required by torchaudio 2.11 `load()`.

## Architecture
```
nova.py                 ← primary entrypoint (argparse CLI: enroll/verify/run)
nova/__main__.py        ← alternate entrypoint (`uv run nova`), simpler, no --duration flag
nova/
  speaker.py            WeSpeaker wrapper (enroll, verify, voiceprint I/O)
  stt.py                faster-whisper transcription
  llm.py                DeepSeek v4-flash via OpenAI client
  tts.py                Kokoro TTS — hardcoded binary path (see below)
  audio.py              Mic capture + Silero VADIterator utterance detection
  wakeword.py           openWakeWord "hey mycroft" detection
  pipeline.py           Main loop: wake → capture → speaker+STT(parallel) → LLM → TTS
```

## Hardcoded paths
- **`nova/tts.py:5`** — `KOKORO_BIN` is hardcoded to `/Users/rawhad/1_Projects/personal_projects/kokoro-rust/target/release/kokoro-tts`. Will break on any machine other than Rohan's. The `pyproject.toml` dependency on `kokoro-tts` (Python wrapper) is NOT used at runtime — the TTS module calls a Rust binary via `subprocess`.

## Testing
- **No test framework.** Tests are `play.py` — manual smoke tests using random noise and tiny models. No pytest, no unittest.
- `play.py` verifies: WeSpeaker model load, faster-whisper load (tiny), Kokoro TTS, LLM client init, Speaker API (enroll/verify), voiceprint save/load roundtrip, UtteranceCapture init.
- To run: `.venv/bin/python play.py`
- Tests use fake audio (numpy random noise), no microphone required.

## Linting / formatting
- **None configured.** No ruff, mypy, black, or pre-commit config exists. Add if needed.

## Docs
- `.dingllm/prds/v1/` — 7 PRD docs covering each component architecture.
- `devlogs.md` — development history, dependency issues resolved, remaining work.
