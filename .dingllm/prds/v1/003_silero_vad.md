# Silero VAD — Deep Dive

> Source: [github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad) | MIT | v6.2.1 | ~260K params

## TL;DR

Silero VAD is a tiny LSTM-based voice activity detection model. It takes 32ms audio chunks (512 samples at 16kHz) and outputs a speech probability [0, 1]. For Nova, we use it to detect when someone stops talking (silence gap of 1.5s = utterance complete).

## Model

- **Architecture**: LSTM with 128-dim hidden state
- **Size**: ~260K params, ~2MB on disk
- **Speed**: <1ms per 32ms chunk on CPU (30x+ realtime)
- **Languages**: Trained on 6000+ languages
- **Runtimes**: PyTorch JIT (default) or ONNX (2-4x faster)
- **Sample rates**: 8kHz or 16kHz

## Installation

```bash
pip install silero-vad          # JIT (PyTorch required)
pip install silero-vad[onnx-cpu]  # ONNX (faster, no PyTorch needed except for torch tensors)
```

## Python API

### Load Model

```python
from silero_vad import load_silero_vad

model = load_silero_vad()          # JIT (default)
model = load_silero_vad(onnx=True)  # ONNX (faster for Apple Silicon)
```

### `get_speech_timestamps()` — Offline (Post-Recording)

Processes an entire audio file and returns speech segments:

```python
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio

model = load_silero_vad()
audio = read_audio("recording.wav", sampling_rate=16000)

speech_timestamps = get_speech_timestamps(
    audio,
    model,
    threshold=0.5,                    # Speech probability threshold
    sampling_rate=16000,
    min_speech_duration_ms=250,       # Ignore short noises
    max_speech_duration_s=float('inf'),  # Max speech segment before forced cut
    min_silence_duration_ms=1500,     # Silence gap to split utterances (KEY PARAM)
    speech_pad_ms=30,                 # Padding on each side
    return_seconds=False,             # False=samples, True=seconds
)

# Returns: [{'start': 12345, 'end': 89012}, ...]
```

### `VADIterator` — Streaming (Real-Time)

Stateful iterator that processes 512-sample chunks sequentially:

```python
from silero_vad import load_silero_vad, VADIterator

model = load_silero_vad()
vad = VADIterator(
    model,
    threshold=0.5,
    sampling_rate=16000,
    min_silence_duration_ms=1500,  # 1.5s silence = utterance end
    speech_pad_ms=30,
)

import torch

for audio_chunk_512samples in mic_stream:
    tensor_chunk = torch.from_numpy(audio_chunk_512samples)
    event = vad(tensor_chunk, return_seconds=True)

    if event:
        if 'start' in event:
            print(f"Speech started at {event['start']}s")
        elif 'end' in event:
            print(f"Utterance complete at {event['end']}s")
            # Process utterance audio (extract from ring buffer)

    # Returns None during ongoing speech or silence (no boundary event)
```

## Parameters for Utterance Segmentation

Key insight: `min_silence_duration_ms` IS the utterance boundary detector. When silence persists for this duration after speech, the segment ends.

| Parameter | Nova Value | Rationale |
|-----------|------------|-----------|
| `threshold` | 0.5 | Speech probability > this = speaking |
| `neg_threshold` | `threshold - 0.15` (auto) | Hysteresis: once in speech, need LOWER prob to exit |
| `min_silence_duration_ms` | **1500** | 1.5s silence gap = utterance complete |
| `min_speech_duration_ms` | 250 | Ignore coughs, clicks, very short noises |
| `speech_pad_ms` | 30 | Small padding to not clip word edges |

## The State Machine (How Boundaries Are Detected)

1. **Not triggered**: If `prob >= threshold` → enter speech, fire `start`
2. **Triggered (in speech)**: If `prob < neg_threshold` → start silence counter
3. **Silence counter >= min_silence** → end speech, fire `end` (utterance complete)
4. **Speech returns before silence threshold** → reset silence counter (brief pause)

The `neg_threshold` hysteresis is critical — once speaking, the model needs a LOWER signal to decide you've stopped (prevents mid-sentence pauses from ending utterances).

## Audio Capture for Utterance Extraction

VADIterator only returns timestamps, NOT audio. To capture utterance audio:

```python
from collections import deque

# 10-second ring buffer of (sample_offset, audio_data) tuples
ring_buffer = deque(maxlen=int(16000 * 10 / 512))

speech_start_offset = None

for chunk in mic_stream:
    ring_buffer.append((current_sample, chunk))

    event = vad(chunk)
    if event:
        if 'start' in event:
            speech_start_offset = event['start']  # In samples
        elif 'end' in event and speech_start_offset is not None:
            # Extract audio from start to end
            utterance = np.concatenate([
                c for (off, c) in ring_buffer
                if speech_start_offset <= off < event['end']
            ])
            # Process utterance...
```

## Gotchas

1. **Chunk size is rigid**: exactly 512 samples (16kHz) or 256 samples (8kHz). Must buffer if PyAudio gives different sizes.
2. **`model.reset_states()` is critical**: LSTM state persists across chunks. Reset between different audio streams/files.
3. **VADIterator doesn't buffer audio**: you must maintain your own ring buffer.
4. **Multi-channel audio**: raises ValueError. Downmix to mono first.
5. **`window_size_samples` is deprecated**: does nothing. Always 512 (16kHz) or 256 (8kHz).
6. **ONNX model opset 15 is 16kHz-only**: use opset 16 for both 8kHz and 16kHz support.
7. **Torchaudio >= 2.9 needs `torchcodec`**: for `read_audio`/`save_audio`. Use `soundfile` as alternative I/O.
8. **macOS MPS**: Model runs on CPU by default. ONNX runtime has Apple Silicon support and is recommended.

## Our Usage for Nova

VAD segmenter running after wake word detection:

```python
# Wake word "hey mycroft" detected → enter listening mode
# Silero VAD captures utterance until 1.5s silence gap
# Returns: audio numpy array for speaker ID + STT processing
```
