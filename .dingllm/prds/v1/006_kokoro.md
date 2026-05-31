# Kokoro TTS — Deep Dive

> Source: [github.com/RohanAwhad/kokoro-tts](https://github.com/RohanAwhad/kokoro-tts) | Thin wrapper around [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)

## TL;DR

Kokoro-tts is a thin Python wrapper around the Kokoro-82M ONNX TTS model. It generates speech from text as a numpy float32 array at 24kHz. 10 voices available (American/British, male/female). Model auto-downloads ~337MB on first use. For Nova: generate TTS audio, write to temp WAV, play via macOS `afplay`.

## Model

- **Backend**: Kokoro-82M ONNX model from hexgrad on HuggingFace
- **Architecture**: ONNX runtime inference
- **Output**: numpy float32 array, 24kHz mono
- **Speed**: ~3.8s to generate 15s of audio on M1 Mac (8 Core)
- **Size**: ~337MB auto-downloaded to `~/.cache/kokoro-tts/kokoro-v0_19.onnx`

## Installation

```bash
# System dependency (required for phonemization)
brew tap justinokamoto/espeak-ng
brew install espeak-ng

# If phonemizer can't find espeak:
export PHONEMIZER_ESPEAK_LIBRARY="/opt/homebrew/bin/espeak-ng"

# Python package
pip install git+https://github.com/RohanAwhad/kokoro-tts.git

# Dependencies: nltk, onnxruntime==1.19.2, phonemizer, requests, torch, soundfile (optional)
```

## Python API

### Basic Usage

```python
from kokoro_tts import Kokoro

kk = Kokoro()
audio = kk.generate("Hello world! I am Nova, your voice assistant.", kk.AVAILABLE_VOICES[0])
# audio: np.ndarray, dtype float32, shape (n_samples,), 24000 Hz
```

### Save to WAV

```python
import soundfile as sf

kk = Kokoro()
audio = kk.generate("Hello world!", "af_bella")
sf.write("output.wav", audio, kk.SAMPLING_RATE)  # SAMPLING_RATE = 24000
```

### Available Voices

`kk.AVAILABLE_VOICES` = list of 10 voice IDs:

| Voice | Description |
|-------|-------------|
| `af` | American Female |
| `af_bella` | American Female (Bella) |
| `af_sarah` | American Female (Sarah) |
| `am_adam` | American Male (Adam) |
| `am_michael` | American Male (Michael) |
| `bf_emma` | British Female (Emma) |
| `bf_isabella` | British Female (Isabella) |
| `bm_george` | British Male (George) |
| `bm_lewis` | British Male (Lewis) |
| `af_nicole` | American Female (Nicole) |
| `af_sky` | American Female (Sky) |

Naming convention: `<accent><gender>[_name]` where `a`=American, `b`=British; `f`=female, `m`=male.

Voice `.pt` files auto-download on first use to `~/.cache/kokoro-tts/voices/`.

## Playback (macOS)

Simplest approach — write WAV to temp file, play with `afplay`:

```python
import soundfile as sf
import subprocess
import tempfile

def speak(text: str, voice: str = "af_bella") -> None:
    kk = Kokoro()
    audio = kk.generate(text, voice)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, kk.SAMPLING_RATE)
        subprocess.run(["afplay", f.name])
        # Cleanup: os.unlink(f.name)
```

Alternative: use `sounddevice` or `pyaudio` for direct playback without temp files.

## Text Processing

- Text is auto-chunked into sentences (max ~25 words each)
- Max 510 tokens per chunk — longer text is truncated with a warning
- `nltk` downloads `punkt_tab` tokenizer on first import

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `nltk` | >=3.9.1 | Sentence tokenization |
| `onnxruntime` | ==1.19.2 | ONNX inference |
| `phonemizer` | >=3.3.0 | Text-to-phoneme (needs espeak-ng) |
| `requests` | >=2.32.3 | Model + voice downloads |
| `torch` | >=2.5.1 | Voice .pt loading |
| `soundfile` | >=0.13.0 | WAV writing (optional, for playback) |

## Gotchas

1. **espeak-ng must be findable** — if phonemizer can't locate it, set `PHONEMIZER_ESPEAK_LIBRARY` env var
2. **First run downloads model** (~337MB) — blocks until complete
3. **First voice use downloads voice file** — auto-cached, small files
4. **No streaming API** — `generate()` returns full array synchronously. Fast enough for voice assistant use (~3.8s for 15s audio).
5. **Text truncation at 510 tokens** — keep LLM responses under ~250 characters
6. **24kHz sample rate** — ensure playback device supports it (macOS does)

## Our Usage for Nova

```python
import soundfile as sf
import tempfile
import subprocess
import os
from kokoro_tts import Kokoro

NOVA_VOICE = "af_bella"  # Choose a voice for Nova's personality

def speak(text: str) -> None:
    kk = Kokoro()
    audio = kk.generate(text, NOVA_VOICE)
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, audio, kk.SAMPLING_RATE)
        subprocess.run(["afplay", path], check=True)
    finally:
        if path and os.path.exists(path):
            os.unlink(path)
```
