# Nova — Voice Assistant PRD

> Created: 2026-05-31 | Status: Planning

## Mission

A sassy voice assistant that knows who it's talking to. Helpful to Rohan. Sassy and unhelpful to everyone else.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Always-listening Loop                    │
│                                                           │
│  Mic → openWakeWord ("hey mycroft") → Wake detected       │
│                                          ↓                │
│                 ┌─────── Silero VAD ──────────┐          │
│                 ↓                             ↓           │
│           Speech frames captured          Silence gap      │
│                 │                           = stop         │
│                 │                                          │
│      ┌──────────┴──────────┐                              │
│      ↓                     ↓                               │
│  Speaker ID (WeSpeaker)  STT (faster-whisper)              │
│  "rohan" / "unknown"     transcript text                   │
│      │                     │                               │
│      └──────────┬──────────┘                               │
│                 ↓                                          │
│      LLM (deepseek-v4-flash, non-thinking)                 │
│      system_prompt: speaker={identity}                     │
│      user_prompt: {transcript}                             │
│                 ↓                                          │
│            TTS (kokoro-tts)                                │
│                 ↓                                          │
│         Play response audio                                │
│                 ↓                                          │
│       Return to listening for "hey mycroft"                │
└─────────────────────────────────────────────────────────┘
```

## Component Stack

| Layer | Library | Details |
|-------|---------|---------|
| Wake word | `openwakeword` | Pre-trained "hey mycroft" model. Custom "Nova" model can be trained later. |
| VAD | `silero-vad` | Bundled with openwakeword. Detects speech vs silence for utterance segmentation. |
| Speaker ID | `wespeaker` | ResNet221_LM English model. Extract embeddings → cosine similarity → threshold. See [001_wespeaker.md](001_wespeaker.md). |
| STT | `faster-whisper` | Local Whisper inference. CPU-optimized via CTranslate2. |
| LLM | DeepSeek v4 Flash | Non-thinking mode. System prompt contains speaker identity. |
| TTS | `kokoro-tts` | Rohan's fork. Kokoro-82M ONNX model. 10 voices, 24kHz output. |
| Audio I/O | `pyaudio` | Mic capture (16kHz mono). `afplay` for playback. |

## Two Modes

| Mode | Command | Description |
|------|---------|-------------|
| Enrollment | `python nova.py enroll` | Records 3-5 speech samples from Rohan. Extracts embeddings via WeSpeaker, averages, L2-normalizes, saves to `~/.nova/voiceprint.npy`. |
| Assistant | `python nova.py` | Starts the always-listening loop. Wake word → VAD → speaker ID + STT (parallel) → LLM (contextual) → TTS → play. |

## Data Flow Details

### 1. Wake Word Detection
- openWakeWord runs continuously on mic stream in 80ms frames
- When "hey mycroft" detected → transition to listening mode
- Future: train custom "Nova" model via Colab notebook (synthetic TTS data, ~1 hour)

### 2. Utterance Capture
- Silero VAD segments speech from silence
- Records until silence gap > 1.5s (configurable)
- Saves utterance to temp WAV buffer

### 3. Speaker ID + STT (Parallel)
- **Speaker ID**: WeSpeaker `extract_embedding()` → cosine similarity against enrolled voiceprint → `True/False` + confidence
- **STT**: faster-whisper transcribes the audio → text
- Both run in separate threads on the same audio buffer

### 4. LLM Call
- System prompt receives `speaker: "Rohan" | "unknown"`
- Rohan → helpful assistant
- Unknown → sassy, contextual refusal

### 5. TTS + Playback
- kokoro generates numpy audio array (24kHz, float32)
- Write to temp WAV via `soundfile`
- Play via `afplay` (macOS built-in)

## LLM Prompt Design

```
system:
You are Nova, a voice assistant with attitude. The person speaking to you has been identified as: {speaker_identity}.

Rules:
- If the speaker is "Rohan": Be helpful, warm, and responsive. Fulfill requests to the best of your ability. Keep responses concise and conversational — this will be spoken aloud.
- If the speaker is NOT "Rohan": You are sassy, witty, and dismissive. Reference their specific request to mock or joke about it, but NEVER actually fulfill it. The goal is to make them laugh at your audacity while leaving empty-handed. Keep responses under 2-3 sentences for voice.

user: {transcript}
```

## Enrollment Flow

```
python nova.py enroll
  → "Please read the following sentence: ..."
  → Record sample 1/5
  → "Please read: ..."
  → Record sample 2/5
  → ... (repeat 5 times)
  → Extract embeddings via WeSpeaker
  → Average → L2-normalize
  → Save to ~/.nova/voiceprint.npy
  → "Enrollment complete. Voiceprint saved."
```

Enrollment best practices (from research):
- 5 samples, 10-30s each, different phrases
- Average embedding centroid
- L2-normalize after averaging
- Threshold: start at cosine similarity > 0.75, calibrate

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| WeSpeaker over SpeechBrain | Built-in `register()`/`recognize()`, better macOS MPS support, state-of-the-art EER (0.447%) |
| kokoro over pyttsx3 | Higher quality voices, 10 voice options, ONNX runtime is fast on M1, Rohan already maintains that fork |
| "hey mycroft" for wake word | Pre-trained model available. "Nova" custom model can be trained and swapped in later. |
| Parallel Speaker ID + STT | Independent models, same audio buffer. No reason to serialize. |
| No wake word for v1 | Lower implementation risk. Wake word refinement is a UI polish task. |

## Reference Docs

- [001_wespeaker.md](001_wespeaker.md) — WeSpeaker deep dive (architecture, API, gaps, our wrapper plan)
- [002_openwakeword.md](002_openwakeword.md) — OpenWakeWord (wake word detection, "hey mycroft" model, custom verifier)
- [003_silero_vad.md](003_silero_vad.md) — Silero VAD (voice activity detection, utterance segmentation, state machine)
- [004_faster_whisper.md](004_faster_whisper.md) — faster-whisper (STT, turbo model, Apple Silicon config)
- [005_deepseek.md](005_deepseek.md) — DeepSeek API (chat completions, thinking mode, prompt design)
- [006_kokoro.md](006_kokoro.md) — Kokoro TTS (voice generation, playback, voice selection)

## Implementation Plan

### Phase 1: Core Pipeline
- [ ] Dependencies & venv setup
- [ ] Enrollment flow (`nova.py enroll`)
- [ ] WeSpeaker wrapper (multi-enrollment, threshold, persistence)
- [ ] STT integration (faster-whisper)
- [ ] LLM integration (deepseek-v4-flash)

### Phase 2: TTS & Audio
- [ ] kokoro TTS integration
- [ ] Audio playback via afplay
- [ ] Mic capture via pyaudio
- [ ] Silero VAD for utterance segmentation

### Phase 3: Always-Listening Loop
- [ ] Wake word detection (openWakeWord "hey mycroft")
- [ ] Main event loop (wake → listen → process → speak → repeat)
- [ ] Concurrency (speaker ID + STT in parallel)
- [ ] Error handling & edge cases

### Phase 4: Polish
- [ ] Voice selection for kokoro
- [ ] Threshold calibration (test with Rohan + other voices)
- [ ] Custom "Nova" wake word model training
- [ ] Logging & debugging utilities
