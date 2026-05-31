# WeSpeaker — Deep Dive

> Source: [github.com/wenet-e2e/wespeaker](https://github.com/wenet-e2e/wespeaker) | Apache 2.0 | Paper: [ICASSP 2023](https://arxiv.org/abs/2210.17016)

## TL;DR

WeSpeaker is a **speaker embedding extraction toolkit**, not a full speaker recognition system. The `register()`/`recognize()` methods are demo-level — they do raw cosine similarity with NO impostor rejection. For Nova, we build on top of `extract_embedding()` and add our own threshold logic, multi-enrollment averaging, and embedding persistence.

## How It Works (High Level)

```
Audio file → torchaudio.load() → [optional: Silero VAD] → Resample 16kHz
    → 80-dim fbank features + CMN → Model forward pass → 256-dim embedding
```

## Key API Surface

### Register (Enrollment)

```python
model = wespeaker.load_model('english')
model.register('rohan', 'sample.wav')   # extracts embedding, stores in-memory
```

**Internal flow (`wespeaker/cli/speaker.py:159-162`):**
1. `torchaudio.load()` → PCM tensor
2. (if `apply_vad=True`) Silero VAD strips silence
3. Resample to 16kHz
4. `torchaudio.compliance.kaldi.fbank()` → 80-dim mel filterbanks
5. CMN (subtract mean over time)
6. Model forward → 256-dim embedding tensor
7. Store in `self.table[name] = embedding`

**Critical**: Stores single embedding per name. No averaging. Duplicate names silently ignored.

### Recognize (Verification)

```python
result = model.recognize('test.wav')
# → {'name': 'rohan', 'confidence': 0.87}
```

**Internal flow (`wespeaker/cli/speaker.py:164-175`):**
1. Extract query embedding (same pipeline as register)
2. Brute-force scan all registered speakers
3. Cosine similarity: `dot(e1, e2) / (|e1| * |e2|)` → mapped to `[0, 1]`
4. Returns argmax speaker + confidence

### Extract Embedding (Low-level — what we actually use)

```python
emb = model.extract_embedding('audio.wav')         # from file
emb = model.extract_embedding_from_pcm(pcm, sr)    # from raw numpy/tensor
emb = model.compute_similarity('a.wav', 'b.wav')   # pairwise score
```

## What `recognize()` Does NOT Do (Critical Gaps)

| Gap | Detail |
|-----|--------|
| **No impostor detection** | Always returns a match, even for strangers. Returns closest registered speaker with whatever confidence. |
| **No threshold** | No `unknown` class. You must implement `confidence < THRESHOLD → "unknown"` yourself. |
| **No score normalization** | Raw cosine only. Formal eval pipeline uses AS-Norm/Z-Norm with cohort sets. |
| **No embedding centering** | No mean subtraction before comparison. |
| **Single enrollment** | One embedding per speaker name. No multi-sample averaging. |
| **VAD off by default** | `self.apply_vad = False`. Silence degrades embedding quality. |
| **Linear scan only** | O(N) per query. Fine for 1-10 speakers. |
| **All in memory** | No disk persistence of embeddings. Re-register on every restart. |
| **Silent crash on silence** | If VAD finds all-silence, returns `None`. `recognize()` crashes. |

## What WeSpeaker DOES Well

- **Embedding quality**: ResNet221_LM achieves 0.447% EER on VoxCeleb1-O
- **Model size**: ~50 MB, auto-downloads to `~/.wespeaker/`
- **macOS MPS**: `model.set_device('mps')` for Apple Silicon
- **Built-in VAD**: Silero VAD integration (`model.set_vad(True)`)
- **Multiple architectures**: ResNet (18-293), ECAPA-TDNN, CAM++, ERes2Net, etc.
- **10+ pretrained models**: English, Chinese, multi-language

## Model Architecture (english/ResNet221_LM)

- **Input**: 80-dim log-mel filterbank (25ms frame, 10ms shift, hamming window)
- **Backbone**: Modified ResNet221 (3x3 kernel, no maxpool, 1-channel)
- **Pooling**: TSTP (Temporal Statistics Pooling — concat(mean, std))
- **Output**: 256-dim speaker embedding
- **Training**: VoxCeleb + additive margin softmax (LM) loss

## Our Usage Pattern for Nova

We build a thin wrapper around WeSpeaker that fixes the gaps:

```python
# Enrollment (our wrapper)
def enroll_rohan(samples: list[str]) -> np.ndarray:
    embeddings = [model.extract_embedding(s) for s in samples]
    averaged = np.mean(embeddings, axis=0)
    return averaged / np.linalg.norm(averaged)  # L2-normalize

# Verification (our wrapper)
def is_rohan(audio_path: str, enrolled: np.ndarray) -> tuple[bool, float]:
    emb = model.extract_embedding(audio_path)
    similarity = np.dot(emb.flatten(), enrolled.flatten())
    return similarity >= THRESHOLD, similarity
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Model inference |
| `torchaudio` | Audio loading + fbank extraction |
| `silero-vad` | Voice Activity Detection (optional) |
| `modelscope` | Model download hub |

## Installation

```bash
pip install git+https://github.com/wenet-e2e/wespeaker.git
```

First-time download: ~50 MB model to `~/.wespeaker/voxceleb_resnet221_LM/`

## Model Files

```
~/.wespeaker/voxceleb_resnet221_LM/
├── avg_model.pt      # Trained checkpoint (~50 MB)
├── config.yaml       # Model config (type, dimensions, pooling)
├── fbank.conf        # Fbank extraction config
└── ...
```

## Gotchas

1. **Model download from modelscope.cn** (Chinese CDN) — may be slow. Mirrors may exist on HuggingFace.
2. **MPS support** works but untested officially. CPU fallback is reliable.
3. **`extract_embedding()` expects 16kHz** — resampling happens automatically but adds latency.
4. **No explicit audio format doc** — relies on `torchaudio`, which supports WAV/FLAC/MP3 (`soundfile` backend).
5. **Embeddings are NOT L2-normalized** by the model — we must normalize after averaging.
