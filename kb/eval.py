"""评测：用一份带标注的问题集衡量检索质量。

指标说明：
  Recall@k —— 期望文档有没有出现在前 k 条里（有没有找到）
  MRR      —— 期望文档的平均倒数名次（排得靠不靠前）
  KW 覆盖  —— 期望关键词在召回上下文里的出现比例（能不能答全）

没有这三个数字，"我优化了检索"就是一句空话。
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import EVAL_SET, TOP_K
from .embed import get_embedder
from .retrieve import Hit, search
from .store import load


def first_rank(hits: list[Hit], expected: set[str]) -> int:
    for rank, hit in enumerate(hits, 1):
        if hit.chunk.path in expected or hit.chunk.doc_id in expected:
            return rank
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="评测检索质量")
    parser.add_argument("-k", type=int, default=TOP_K)
    parser.add_argument("--md", action="store_true", help="输出 Markdown 表格")
    args = parser.parse_args()

    if not EVAL_SET.exists():
        print(f"缺少评测集: {EVAL_SET}")
        return 1
    rows = [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    index = load(get_embedder())

    found, rr, kw_all, kw_hit = 0, 0.0, 0, 0
    detail = []
    for row in rows:
        hits = search(index, row["question"], k=args.k)
        expected = set(row.get("expected_docs") or [])
        rank = first_rank(hits, expected) if expected else 0
        found += 1 if rank else 0
        rr += 1.0 / rank if rank else 0.0
        context = "\n".join(hit.chunk.text for hit in hits)
        keywords = row.get("expected_keywords") or []
        got = sum(1 for kw in keywords if kw in context)
        kw_all += len(keywords)
        kw_hit += got
        detail.append((row["question"], rank, got, len(keywords)))

    n = len(rows)
    recall = found / n if n else 0.0
    mrr = rr / n if n else 0.0
    coverage = kw_hit / kw_all if kw_all else 0.0

    if args.md:
        print("| 问题 | 期望文档名次 | 关键词命中 |")
        print("| --- | --- | --- |")
        for question, rank, got, total in detail:
            print(f"| {question[:28]} | {rank or '—'} | {got}/{total} |")
        print()
        print(f"**Recall@{args.k}** = {recall:.2f}　**MRR** = {mrr:.2f}　**KW 覆盖** = {coverage:.0%}")
        return 0

    for question, rank, got, total in detail:
        mark = "✓" if rank else "✗"
        print(f"{mark} r={rank or '-':<2} kw={got}/{total}  {question[:40]}")
    print(f"\n问题数 {n}")
    print(f"Recall@{args.k}: {recall:.2%}")
    print(f"MRR:        {mrr:.3f}")
    print(f"KW 覆盖:    {coverage:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
