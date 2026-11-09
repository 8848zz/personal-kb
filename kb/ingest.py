"""建索引：遍历 data/raw，切分、脱敏、向量化、落盘。

分级规则：路径里含 L0 目录的视为可上云（公开资料），其余一律 L1 本地。
命中拒绝清单的文件（密码、证件等）直接跳过，并在汇总里点名。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .chunk import split
from .config import RAW, TIER_CLOUD, ensure_dirs
from .embed import get_embedder
from .store import is_denied, save
from .textio import SUPPORTED, read_text


def tier_of(path: Path) -> str:
    return TIER_CLOUD if "l0" in {p.lower() for p in path.parts} else "L1"


def collect() -> tuple[list, list[tuple[Path, str]]]:
    chunks, skipped = [], []
    for path in sorted(p for p in RAW.rglob("*") if p.is_file()):
        if path.suffix.lower() not in SUPPORTED:
            continue
        if is_denied(path):
            skipped.append((path, "命中拒绝清单（疑似密码/证件），拒绝入库"))
            continue
        try:
            text = read_text(path)
        except Exception as exc:
            skipped.append((path, f"读取失败: {exc}"))
            continue
        rel = path.relative_to(RAW).as_posix()
        pieces = split(text, doc_id=path.stem, path=rel, tier=tier_of(path))
        if not pieces:
            skipped.append((path, "切分后为空"))
            continue
        chunks.extend(pieces)
    return chunks, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="建立个人档案索引")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不向量化")
    args = parser.parse_args()

    ensure_dirs()
    chunks, skipped = collect()

    if not chunks:
        print(f"没有可入库的内容。把资料放进 {RAW} 下（可按 L0 / L1 分子目录）。")
        return 1

    docs = len({c.path for c in chunks})
    print(f"文件 {docs} 个 → 片段 {len(chunks)} 条")
    for path, reason in skipped:
        print(f"  跳过 {path.name}: {reason}")

    if args.dry_run:
        return 0

    embedder = get_embedder()
    print(f"向量化中（{embedder.name}）...", end=" ", flush=True)
    vectors = embedder.embed([c.embed_text for c in chunks])
    save(chunks, vectors, embedder.name)
    print(f"完成，维度 {len(vectors[0])}，已写入 data/index/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
