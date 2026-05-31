# OpenWakeWord — Deep Dive

> Source: [github.com/dscripka/openWakeWord](https://github.com/dscripka/openWakeWord) | Apache 2.0 | Pre-trained models: CC BY-NC-SA 4.0

## TL;DR

OpenWakeWord is a lightweight wake word detection library. It bundles a shared speech embedding model + per-wakeword classifiers. Pre-trained models exist for "alexa", "hey mycroft", "hey jarvis", "hey rhasspy", "weather", and "timer". Custom models can be trained via Colab notebook (~1 hour, synthetic data only). Includes optional Silero VAD for false-positive reduction.

For Nova: we use the pre-trained "hey mycroft" model as a placeholder. Train custom "Nova" model later.

## Architecture

```
16-bit 16kHz PCM audio (80ms frames)
    → Melspectrogram (ONNX)
    → 96-dim speech embeddings (frozen Google speech_embedding backbone)
    → Wake word classifier (small FC/RNN head)
    → Score [0, 1]
```

Three separate ONNX/TFLite models compose: melspectrogram preprocessor, shared embedding backbone, per-wakeword classifier heads. The embedding backbone is frozen (not trained) and shared across all wake word models.

## Python API

### Model Constructor

```python
from openwakeword.model import Model

model = Model(
    wakeword_models=["hey_mycroft"],   # names or paths to .tflite/.onnx files
    vad_threshold=0.0,                 # 0=disabled. 0.3-0.7 for VAD filtering
    inference_framework="tflite",      # "tflite" or "onnx"
    enable_speex_noise_suppression=False,  # Linux only! Crashes on macOS
)
```

### `predict(x, **kwargs) -> dict[str, float]`

```python
# x: np.ndarray, int16, multiples of 1280 samples (80ms) recommended
prediction = model.predict(audio_chunk)

# Returns: {"hey_mycroft": 0.02, "alexa": 0.87, ...}
# First 5 frames always return 0.0 (warm-up)
```

### Patience & Debounce

```python
# Patience: N consecutive frames above threshold required
prediction = model.predict(audio, patience={"hey_mycroft": 3}, threshold={"hey_mycroft": 0.5})

# Debounce: suppress re-activations for N seconds (MUTUALLY EXCLUSIVE with patience)
prediction = model.predict(audio, debounce_time=2.0, threshold={"hey_mycroft": 0.5})
```

### Reset

```python
model.reset()  # Clear prediction + feature buffers
```

## Continuous Mic Detection Pattern

```python
import pyaudio
import numpy as np
from openwakeword.model import Model

CHUNK = 1280  # 80ms @ 16kHz (must be multiples of 1280)
RATE = 16000

model = Model(wakeword_models=["hey_mycroft"], inference_framework="tflite")

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

while True:
    pcm = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    prediction = model.predict(pcm)
    for name, score in prediction.items():
        if score > 0.5:
            print(f"WOKE: {name} ({score:.3f})")
```

## "hey mycroft" Model

- **Architecture**: Flatten(1536) → Linear(64) → LayerNorm → ReLU → Linear(64) → LayerNorm → ReLU → Linear(1) → Sigmoid
- **Trainable params**: ~103K
- **Training data**: ~100K synthetic clips (WaveGlow + VITS) + ~31K hours negative data
- **Expected perf**: <5% false-reject, <0.5 false-accept/hour at threshold ~0.5
- **Robustness**: Handles whisper, varied speed, some accent variation

## VAD Integration (Bundled)

OpenWakeWord bundles Silero VAD for false-positive suppression:

```python
model = Model(wakeword_models=["hey_mycroft"], vad_threshold=0.5)
```

When enabled, wake word predictions are zeroed out if VAD score < threshold in the 0.4-0.56s window before the current frame. This reduces false activations from non-speech noise.

## Custom Verifier Models (Speaker-Specific Wake Word)

OpenWakeWord supports a 2nd-stage **speaker-specific verifier** that only allows activations if the wake word was spoken by a known voice:

```python
from openwakeword import train_custom_verifier

train_custom_verifier(
    positive_reference_clips=["my_hey_mycroft_1.wav", "my_hey_mycroft_2.wav", "my_hey_mycroft_3.wav"],
    negative_reference_clips=["my_other_speech.wav"],
    output_path="rohan_verifier.pkl",
    model_name="hey_mycroft",
)
```

Then use it:
```python
model = Model(
    wakeword_models=["hey_mycroft"],
    custom_verifier_models={"hey_mycroft": "rohan_verifier.pkl"},
    custom_verifier_threshold=0.3,
)
```

**Note**: This is useful but NOT the same as full speaker ID. The verifier is a logistic regression on the 96-dim embeddings from the wake word context only. For general speaker verification, we still need WeSpeaker.

## Gotchas

1. **Speex noise suppression is Linux-only** — setting `enable_speex_noise_suppression=True` crashes on macOS
2. **First 5 frames always return 0.0** — intentional warm-up, accounts for ~400ms
3. **Audio MUST be int16** — passing float32 raises ValueError
4. **Chunk size should be multiples of 1280** (80ms) — non-multiples add latency
5. **Patience and debounce are mutually exclusive** — ValueError if both set
6. **Pre-trained models are CC BY-NC-SA 4.0** — non-commercial only. Code is Apache 2.0.
7. **No acoustic echo cancellation** — playing TTS while mic is active degrades performance
8. **macOS tflite uses `ai-edge-litert`** — fallback to `inference_framework="onnx"` if install fails

## Our Usage for Nova

```python
# Wake word detection runs continuously on mic stream
# When "hey mycroft" score > 0.5 → transition to listening mode
# Future: swap to custom "Nova" model

model = Model(wakeword_models=["hey_mycroft"], vad_threshold=0.3, inference_framework="tflite")
```
