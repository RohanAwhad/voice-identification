# faster-whisper — Deep Dive

> Source: [github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT | CTranslate2 backend

## TL;DR

faster-whisper is a reimplementation of OpenAI's Whisper using CTranslate2 (C++ inference engine). 4x faster and uses less memory than openai/whisper. Supports all Whisper models + distil-whisper + turbo variants. For Nova: runs on macOS CPU, int8 quantization, transcribes short utterances from numpy arrays.

## Installation

```bash
pip install faster-whisper
# Dependencies: ctranslate2, huggingface_hub, onnxruntime, av (PyAV — no FFmpeg needed)
```

No system dependencies needed. PyAV bundles FFmpeg. macOS works out of the box.

## Model Sizes (Relevant to Voice Assistant)

| Model | Params | Download | RAM (int8) | Speed (M1) | Accuracy |
|-------|--------|----------|------------|------------|----------|
| `tiny` | 39M | ~75 MB | ~500 MB | 20-30x realtime | Lowest |
| `base` | 74M | ~145 MB | ~700 MB | ~15x | Low |
| `small` | 244M | ~488 MB | ~1.2 GB | ~6-10x | Medium |
| `medium` | 769M | ~1.5 GB | ~2.5 GB | ~2-4x | Good |
| **`turbo`** | **~809M** | **~1.6 GB** | **~2.8 GB** | **~4-8x** | **Near large-v3** |
| `large-v3` | 1550M | ~3.1 GB | ~4 GB | ~1-2x | Highest |

**Recommendation for Nova**: `"turbo"` with `compute_type="int8"`. Best speed/accuracy tradeoff for Apple Silicon.

## Apple Silicon Critical Note

**DO NOT use `compute_type="float16"`** on Apple Silicon. CTranslate2 has NO optimized float16 execution on AArch64 — it silently falls back to float32 but still loads FP16 weights from disk (wasted bandwidth).

**Use `compute_type="int8"`** — weights are quantized to 8-bit, computation runs at float32. ~2x faster, ~50% less RAM.

## Python API

### WhisperModel Constructor

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    model_size_or_path: str,      # "turbo", "small", "medium", "large-v3", etc.
    device: str = "auto",         # "cpu" or "cuda" (no CUDA on macOS = "cpu")
    compute_type: str = "default",# "int8" for Apple Silicon
    cpu_threads: int = 0,         # 0 = OMP_NUM_THREADS env var
    download_root: str = None,    # Custom cache dir
)
```

### `transcribe()` — Core Method

```python
segments, info = model.transcribe(
    audio: Union[str, BinaryIO, np.ndarray],  # File path, file object, or numpy array
    language: str = None,                      # "en", "fr", etc. None = auto-detect
    beam_size: int = 5,                        # Reduce to 1 for speed
    vad_filter: bool = False,                  # Enable built-in Silero VAD
    vad_parameters: dict = None,               # Customize VAD behavior
    without_timestamps: bool = False,           # Text-only (no timing data)
    condition_on_previous_text: bool = True,   # False for independent utterances
    temperature: ... = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    # ... many more params
)

# Returns: (generator[Segment], TranscriptionInfo)
```

**`Segment` dataclass:**
```python
@dataclass
class Segment:
    id: int               # Segment index
    start: float          # Start time (seconds)
    end: float            # End time (seconds)
    text: str             # Transcribed text
    avg_logprob: float    # Average log probability
    no_speech_prob: float # Probability of no speech (= VAD score)
```

**`TranscriptionInfo` dataclass:**
```python
@dataclass
class TranscriptionInfo:
    language: str                    # Detected language code
    language_probability: float      # Confidence
    duration: float                  # Audio duration
    duration_after_vad: float        # Duration after VAD trimming
```

**IMPORTANT**: `segments` is a **generator**. Transcription only starts when you iterate it.

### Pattern for Short Utterances

```python
model = WhisperModel("turbo", device="cpu", compute_type="int8")

def transcribe_utterance(audio: np.ndarray) -> str:
    """audio: float32, 16kHz mono numpy array"""
    segments, info = model.transcribe(
        audio,
        beam_size=1,
        vad_filter=False,           # We already VAD'd externally
        condition_on_previous_text=False,  # Don't carry context
    )
    # Force iteration for short audio
    return " ".join(seg.text.strip() for seg in segments)
```

## Audio Input Requirements

| Input type | Requirements |
|-----------|--------------|
| File path / file object | Any format PyAV supports (WAV, MP3, FLAC, etc.). Auto-resampled to 16kHz mono. |
| Numpy array | **Must be 16kHz mono**. dtype float32, values in [-1, 1]. NO auto-resample for numpy input. |

If you pass numpy, YOU are responsible for getting the sample rate right. Wrong sample rate = wrong timestamps + degraded accuracy.

## Built-in VAD (via Silero VAD ONNX)

faster-whisper bundles Silero VAD for silence removal before transcription:

```python
segments, info = model.transcribe(
    audio,
    vad_filter=True,
    vad_parameters=dict(
        threshold=0.5,
        min_silence_duration_ms=500,   # Split on 500ms silence
        min_speech_duration_ms=250,    # Filter short noises
        speech_pad_ms=200,
    ),
)
```

Note: We're already using external Silero VAD for utterance segmentation, so we likely DON'T need built-in VAD too. But it's useful as a fallback.

## Gotchas

1. **First run downloads model** — blocks until download completes (~75MB-3GB to `~/.cache/huggingface/hub/`)
2. **segments is a generator** — `transcribe()` returns immediately. Actual transcription happens on iteration.
3. **Numpy input must be 16kHz** — no resampling for numpy. Only file input auto-resamples.
4. **float16 = wasted on Apple Silicon** — CTranslate2 falls back to float32. Always use `"int8"`.
5. **turbo model repo**: The canonical HF repo may have changed. Use `WhisperModel("Systran/faster-whisper-large-v3-turbo")` for explicit path.
6. **`condition_on_previous_text=False`** — important for independent utterances. Otherwise previous transcript leaks context.
7. **Beam size 5 is default** — reduce to 1 for voice assistant speed.
8. **Temperature fallback**: Default tries [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]. If one temp produces gibberish (high compression_ratio), it tries the next.

## Our Usage for Nova

```python
model = WhisperModel("turbo", device="cpu", compute_type="int8")

# Audio comes from Silero VAD utterance extraction (float32, 16kHz, mono numpy array)
segments, info = model.transcribe(
    utterance_audio,
    beam_size=1,
    language="en",
    vad_filter=False,
    condition_on_previous_text=False,
)
transcript = " ".join(seg.text.strip() for seg in segments)
```
