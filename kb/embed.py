"""向量化。默认走本地 Ollama；KB_OFFLINE=1 时改用哈希向量（仅用于验证管道）。

隐私要点：embedding 决定了「哪些文本会被送出本机」。
敏感层一定要用本地模型，否则原文在建索引阶段就已经上云了。
"""
from __future__ import annotations

import hashlib
import json
import urllib.request

from .config import EMBED_MODEL, OFFLINE, OLLAMA_URL
from .tokenize import tokenize


def l2(vec: list[float]) -> list[float]:
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm else vec


def _post(path: str, payload: dict, timeout: int = 180) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL + path,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class HashEmbedder:
    """无模型、确定性的散列向量。只为在没装 embedding 模型时跑通流程。"""

    dim = 512
    name = "hash-512"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = int.from_bytes(
                hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "little"
            )
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return l2(vec)


class OllamaEmbedder:
    def __init__(self, model: str = EMBED_MODEL, batch: int = 8):
        self.model = model
        self.batch = batch
        self.name = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            piece = texts[i : i + self.batch]
            data = _post("/api/embed", {"model": self.model, "input": piece})
            vectors = data.get("embeddings")
            if not vectors:
                raise RuntimeError(f"Ollama 未返回向量: {data}")
            out.extend(l2(v) for v in vectors)
        return out


def get_embedder():
    return HashEmbedder() if OFFLINE else OllamaEmbedder()
