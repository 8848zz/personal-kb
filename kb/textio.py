"""读取原料文件。PDF 用 pypdf（可选依赖），其余按纯文本读。"""
from __future__ import annotations

import re
from pathlib import Path

SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}

# 入库前的脱敏：手机号、身份证、银行卡
_REDACT = (
    (re.compile(r"\b1[3-9]\d{9}\b"), "[手机号已脱敏]"),
    (re.compile(r"\b\d{17}[\dXx]\b"), "[身份证已脱敏]"),
    (re.compile(r"\b\d{16,19}\b"), "[卡号已脱敏]"),
)


def redact(text: str) -> str:
    for pattern, tag in _REDACT:
        text = pattern.sub(tag, text)
    return text


def read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    return redact(raw)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"读取 {path.name} 需要 pypdf，请先执行: pip install pypdf"
        ) from exc
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"## 第 {i} 页\n\n{text.strip()}")
    return redact("\n\n".join(pages))
