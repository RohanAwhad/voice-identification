# Nova — Voice Assistant

Sassy voice assistant with speaker identification. Helpful to Rohan, sassy to everyone else.

## Setup

```
brew install espeak-ng
uv venv --python $(brew --prefix python@3.12)/bin/python3.12 .venv
source .venv/bin/activate
uv pip install -e . "git+https://github.com/wenet-e2e/wespeaker.git" "git+https://github.com/RohanAwhad/kokoro-tts.git"
uv pip install torchcodec

# Patch s3prl for torchaudio 2.0+
sed -i '' 's/torchaudio.set_audio_backend("sox_io")/# torchaudio.set_audio_backend("sox_io") /' .venv/lib/python3.12/site-packages/s3prl/upstream/byol_s/byol_a/common.py
sed -i '' 's/from torchaudio.sox_effects import apply_effects_tensor/# from torchaudio.sox_effects import apply_effects_tensor /' .venv/lib/python3.12/site-packages/s3prl/upstream/mos_prediction/expert.py
```

Set your DeepSeek API key:

```
export DEEPSEEK_API_KEY=sk-...
```

## Quick Test

```
.venv/bin/python play.py
```

## Usage

```
.venv/bin/python nova.py enroll                    # Record 5 voice samples, create voiceprint
.venv/bin/python nova.py verify                    # Test if current speaker matches voiceprint (5s)
.venv/bin/python nova.py verify --duration 10      # Test with 10s recording
.venv/bin/python nova.py run                       # Start always-listening assistant loop
```

### Enrollment
Records 5 speech samples (8s each). Reads prompts from screen. Saves voiceprint to `~/.nova/voiceprint.npy`.

### Verify
Records speech and checks against saved voiceprint. Prints identity + confidence score.

### Run
Always-listening loop. Say "hey mycroft" to wake, then speak your request.

## Architecture

```
Mic → openWakeWord ("hey mycroft") → Silero VAD ─┬─> WeSpeaker (speaker ID)
                                                  └─> faster-whisper (STT)
                                                            ↓
                                                   DeepSeek v4-flash
                                                            ↓
                                                     Kokoro TTS
                                                            ↓
                                                      afplay
```

## Stack

| Layer | Library | Details |
|-------|---------|---------|
| Wake word | openWakeWord | "hey mycroft" placeholder model |
| VAD | silero-vad | 1.5s silence gap = utterance boundary |
| Speaker ID | WeSpeaker | ResNet221_LM, cosine similarity > 0.75 |
| STT | faster-whisper | turbo model, int8 on Apple Silicon |
| LLM | DeepSeek v4-flash | Non-thinking mode, 256 max tokens |
| TTS | Kokoro-82M | af_bella voice, 24kHz, afplay playback |
