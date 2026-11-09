"""分词：不装 jieba，用「ASCII 词 + 汉字单字 + 汉字二元组」。

为什么够用：中文检索里，二元组（bigram）几乎等价于粗粒度分词，
对 BM25 来说已经能区分大部分文档。好处是零依赖、无词典、对新词免疫。
"""
from __future__ import annotations

import re

_ASCII_WORD = re.compile(r"[a-z0-9_]{2,}")
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    out: list[str] = _ASCII_WORD.findall(text)
    cjk = _CJK_CHAR.findall(text)
    out.extend(cjk)
    out.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return out


def is_denied(name: str) -> bool:
    """文件名命中拒绝清单则为 True（L2 绝密，绝不入库）。"""
    from .config import DENY_PATTERNS

    low = name.lower()
    return any(p in low for p in DENY_PATTERNS)
