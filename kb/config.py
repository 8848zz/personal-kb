"""全局配置。所有可调项集中在这里，代码里不出现魔法数字。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = Path(os.environ.get("KB_RAW", DATA / "raw"))  # 原料目录；KB_RAW=data/demo 用虚构样例
INDEX = DATA / "index"      # 索引产物
EVAL_SET = DATA / "eval" / "questions.jsonl"

# ---- Ollama ----
OLLAMA_URL = os.environ.get("KB_OLLAMA", "http://127.0.0.1:11434")
GEN_MODEL = os.environ.get("KB_GEN_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.environ.get("KB_EMBED_MODEL", "bge-m3")

# 离线模式：不调 Ollama 的 embedding，改用本地哈希向量。
# 只用于验证管道是否通畅，检索质量会明显下降。
OFFLINE = os.environ.get("KB_OFFLINE", "0") == "1"

# ---- 检索 ----
TOP_K = 5
RRF_K = 60           # Reciprocal Rank Fusion 的平滑常数

# ---- 切分 ----
CHUNK_MAX = 700      # 单个片段最大字符数（中文按字符算，约等于 400 token）
CHUNK_MIN = 120      # 低于这个长度的片段会被丢弃，避免碎片污染检索

# ---- 分级：L0 可上云，L1 必须本地，L2 禁止入库 ----
TIER_CLOUD = "L0"
TIER_LOCAL = "L1"

# 命中这些关键词的文件一律拒绝入库（L2 绝密）
DENY_PATTERNS = (
    "密码", "password", "passwd", "secret", "token",
    "身份证", "idcard", "id_card", "银行卡", "bankcard",
    "私钥", "private_key", "助记词",
)


def ensure_dirs() -> None:
    for p in (RAW, INDEX, EVAL_SET.parent):
        p.mkdir(parents=True, exist_ok=True)
