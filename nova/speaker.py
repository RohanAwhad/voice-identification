from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
import torch
import wespeaker


class SpeakerIdentity(Enum):
    ROHAN = "Rohan"
    UNKNOWN = "unknown"


DEFAULT_THRESHOLD = 0.75
N_SAMPLES = 5
SAMPLE_LENGTH_S = 8

ENROLL_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump.",
    "The five boxing wizards jump quickly.",
    "Sphinx of black quartz, judge my vow.",
]


def _to_numpy(emb: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(emb, torch.Tensor):
        return emb.detach().cpu().numpy()
    return emb


class SpeakerVerifier:
    def __init__(self, use_vad: bool = True) -> None:
        self.model = wespeaker.load_model("english")
        if use_vad:
            self.model.set_vad(True)

    def enroll(self, samples: list[np.ndarray]) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for audio in samples:
            pcm = torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0)
            emb = self.model.extract_embedding_from_pcm(pcm, 16000)
            embeddings.append(_to_numpy(emb))

        averaged = np.mean(embeddings, axis=0)
        l2_norm = averaged / np.linalg.norm(averaged)
        return l2_norm.astype(np.float32).flatten()

    def verify(
        self,
        audio: np.ndarray,
        voiceprint: np.ndarray,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> tuple[SpeakerIdentity, float]:
        pcm = torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0)
        emb_raw = self.model.extract_embedding_from_pcm(pcm, 16000)

        if emb_raw is None:
            return SpeakerIdentity.UNKNOWN, 0.0

        emb = _to_numpy(emb_raw).flatten().astype(np.float32)
        voiceprint = voiceprint.flatten().astype(np.float32)
        similarity = float(np.dot(emb, voiceprint))

        identity = SpeakerIdentity.ROHAN if similarity >= threshold else SpeakerIdentity.UNKNOWN
        return identity, similarity


def save_voiceprint(embedding: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embedding)


def load_voiceprint(path: Path) -> np.ndarray:
    return np.load(path)


def default_voiceprint_path() -> Path:
    return Path.home() / ".nova" / "voiceprint.npy"
