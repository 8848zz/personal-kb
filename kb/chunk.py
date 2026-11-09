"""切分：按 Markdown 标题层级切，超长再按段落二次切。

为什么不用固定窗口：固定窗口会把一个论点劈成两半，
导致「检索到了但答不全」。标题感知的切分让每个片段自带语义边界。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import CHUNK_MAX, CHUNK_MIN


@dataclass
class Chunk:
    doc_id: str
    path: str          # 相对 data/raw 的路径，用于在答案里标出处
    heading: str       # 标题路径，如 "第二章 > 2.1 向量检索"
    start_line: int
    text: str
    tier: str = "L1"

    @property
    def embed_text(self) -> str:
        """喂给 embedding 的文本：带上标题，避免片段脱离上下文后失去主题。"""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


def _blocks(lines: list[str]):
    """按标题切块，产出 (标题, 起始行号, 正文)。"""
    heading, start, buf = "", 1, []
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("#"):
            body = "\n".join(buf).strip()
            if body:
                yield heading, start, body
            heading = line.strip().lstrip("#").strip()
            start = i + 1
            buf = []
        else:
            buf.append(line)
    body = "\n".join(buf).strip()
    if body:
        yield heading, start, body


def _split_long(text: str, limit: int):
    """按空行分段累积，单段仍然超长时硬切。"""
    cur = ""
    for para in (p.strip() for p in text.split("\n\n") if p.strip()):
        if len(cur) + len(para) + 2 <= limit:
            cur = f"{cur}\n\n{para}".strip()
            continue
        if cur:
            yield cur
        while len(para) > limit:
            yield para[:limit]
            para = para[limit:]
        cur = para
    if cur:
        yield cur


def _merge_small(items: list[list]) -> list[list]:
    """把小片段并进相邻片段。

    注意：不能简单地「丢弃短片段」。个人笔记大量是短条目
    （一个条目两行字），直接丢弃等于把整份档案扔了——
    这里的第一版就踩过这个坑，2 个文件被切成了 0 条。
    """
    out: list[list] = []
    for heading, start, text in items:
        if (
            out
            and len(out[-1][2]) < CHUNK_MIN
            and out[-1][0] == heading
            and len(out[-1][2]) + len(text) + 2 <= CHUNK_MAX
        ):
            out[-1][2] = f"{out[-1][2]}\n\n{text}"
            continue
        out.append([heading, start, text])

    # 第二轮：换标题的小片段并入前一片段，标题沿用前一个（有轻微信息损失）
    merged: list[list] = []
    for heading, start, text in out:
        if (
            merged
            and len(text) < CHUNK_MIN
            and len(merged[-1][2]) + len(text) + 2 <= CHUNK_MAX
        ):
            merged[-1][2] = f"{merged[-1][2]}\n\n{text}"
            continue
        merged.append([heading, start, text])
    return merged


def split(text: str, doc_id: str, path: str, tier: str = "L1") -> list[Chunk]:
    raw: list[list] = []
    for heading, start, body in _blocks(text.splitlines()):
        # start_line 是该标题块正文的绝对行号；二次切分后的片段共用它作近似出处
        for piece in _split_long(body, CHUNK_MAX):
            if piece.strip():
                raw.append([heading, start, piece.strip()])

    return [
        Chunk(
            doc_id=doc_id,
            path=path,
            heading=heading,
            start_line=start,
            text=piece,
            tier=tier,
        )
        for heading, start, piece in _merge_small(raw)
        if len(piece) >= 20
    ]
