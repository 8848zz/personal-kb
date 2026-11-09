"""索引的持久化与加载。

存三件东西：
  chunks.jsonl  —— 片段正文与出处（人可读，方便排查）
  vectors.f32   —— 归一化后的向量，float32 二进制（2000 片段约 8 MB）
  meta.json     —— 维度、条数、用了哪个 embedding 模型
"""
from __future__ import annotations

import json
from array import array
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .bm25 import BM25
from .chunk import Chunk
from .config import INDEX
from .tokenize import tokenize

CHUNKS = INDEX / "chunks.jsonl"
VECTORS = INDEX / "vectors.f32"
META = INDEX / "meta.json"


def save(chunks: list[Chunk], vectors: list[list[float]], embed_model: str) -> None:
    INDEX.mkdir(parents=True, exist_ok=True)
    with CHUNKS.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    flat = array("f", (value for vec in vectors for value in vec))
    with VECTORS.open("wb") as fh:
        flat.tofile(fh)
    meta = {
        "embed_model": embed_model,
        "dim": len(vectors[0]) if vectors else 0,
        "count": len(chunks),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


class Index:
    def __init__(self, chunks, vectors, meta, embedder):
        self.chunks: list[Chunk] = chunks
        self.vectors: list[list[float]] = vectors
        self.meta: dict = meta
        self.embedder = embedder
        self.tokens = [tokenize(c.embed_text) for c in chunks]
        self.bm25 = BM25(self.tokens)

    def __len__(self) -> int:
        return len(self.chunks)


def load(embedder) -> Index:
    if not CHUNKS.exists():
        raise SystemExit("索引不存在。先执行: python -m kb.ingest")
    meta = json.loads(META.read_text(encoding="utf-8"))
    if meta["embed_model"] != embedder.name:
        raise SystemExit(
            f"索引是用 {meta['embed_model']} 建的，当前 embedding 是 "
            f"{embedder.name}。两者维度不同，不能混用。\n"
            f"要么重跑 ingest，要么设置 KB_EMBED_MODEL={meta['embed_model']}。"
        )
    chunks = [
        Chunk(**json.loads(line))
        for line in CHUNKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dim, count = meta["dim"], meta["count"]
    flat = array("f")
    with VECTORS.open("rb") as fh:
        flat.fromfile(fh, count * dim)
    vectors = [flat[i * dim : (i + 1) * dim].tolist() for i in range(count)]
    return Index(chunks, vectors, meta, embedder)


def is_denied(path: Path) -> bool:
    from .tokenize import is_denied as _is_denied

    return _is_denied(path.name)
