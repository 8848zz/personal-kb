"""BM25：关键词侧的排序器，与向量检索互补。

向量擅长「意思相近」，BM25 擅长「字面命中」——专有名词、型号、人名
这类东西向量经常召不回，关键词一定能。所以要混合。
"""
from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.n = len(docs)
        self.lens = [len(d) for d in docs]
        self.avg_len = (sum(self.lens) / self.n) if self.n else 0.0
        self.tfs = [Counter(d) for d in docs]
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query: list[str]) -> list[float]:
        q = Counter(query)
        out = [0.0] * self.n
        for term, qf in q.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(term, 0)
                if not f:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg_len or 1))
                out[i] += idf * qf * f * (self.k1 + 1) / (f + norm)
        return out
