"""混合检索：向量召回 + 关键词召回，用 RRF 融合。

为什么用 RRF 而不是加权求和：两个排序器的分数分布完全不同
（余弦是 0~1，BM25 是无上界），直接相加要先归一化，而归一化本身
会随查询变化。RRF 只看名次，不需要标定，实践中最稳。
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import RRF_K, TOP_K
from .store import Index
from .tokenize import tokenize


@dataclass
class Hit:
    chunk: object
    score: float      # 融合分
    vec_score: float
    bm25_score: float
    rank_vec: int
    rank_bm25: int


def _top(scores: list[float], limit: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]


def _rrf(rank_lists: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + pos)
    return fused


def search(index: Index, query: str, k: int = TOP_K, pool: int = 50) -> list[Hit]:
    qvec = index.embedder.embed([query])[0]
    if len(qvec) != index.meta["dim"]:
        raise SystemExit(
            f"查询向量维度 {len(qvec)} 与索引维度 {index.meta['dim']} 不一致。"
        )
    vec_scores = [
        sum(a * b for a, b in zip(qvec, vec)) for vec in index.vectors
    ]
    bm25_scores = index.bm25.scores(tokenize(query))

    ranks_vec = _top(vec_scores, pool)
    ranks_bm25 = _top(bm25_scores, pool)
    fused = _rrf([ranks_vec, ranks_bm25])

    pos_vec = {idx: r for r, idx in enumerate(ranks_vec, 1)}
    pos_bm25 = {idx: r for r, idx in enumerate(ranks_bm25, 1)}

    hits = [
        Hit(
            chunk=index.chunks[idx],
            score=score,
            vec_score=vec_scores[idx],
            bm25_score=bm25_scores[idx],
            rank_vec=pos_vec.get(idx, 0),
            rank_bm25=pos_bm25.get(idx, 0),
        )
        for idx, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    ]
    return hits
