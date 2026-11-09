"""提问：检索 → 拼上下文 → 本地模型作答 → 打印出处。

答案的每个结论都必须带 [编号]，编号对应下面的来源列表。
找不到就直说找不到——这是「知识库」和「算命」的分界线。
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

from .config import GEN_MODEL, OLLAMA_URL, TOP_K
from .embed import get_embedder
from .retrieve import Hit, search
from .store import load

PROMPT = """你是个人档案库的问答助手。只能依据下面提供的资料回答。

规则：
1. 只使用资料中的信息，不要补充资料以外的知识。
2. 每个结论后面用 [编号] 标注出处，例如「绩点是 3.7 [2]」。
3. 如果资料里找不到答案，直接说「资料里没有找到」，不要猜测或编造。
4. 不要对提问者的性格、能力下判断，除非资料里有明确记录。
5. 用中文回答，简洁，不要复述规则。

资料：
{context}

问题：{question}"""


def render(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, 1):
        c = hit.chunk
        blocks.append(f"[{i}] 出处: {c.path} › {c.heading}（第 {c.start_line} 行）\n{c.text}")
    return "\n\n".join(blocks)


def stream(messages: list[dict]) -> str:
    payload = {
        "model": GEN_MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }
    req = urllib.request.Request(
        OLLAMA_URL + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    parts: list[str] = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            obj = json.loads(line)
            piece = (obj.get("message") or {}).get("content", "")
            if piece:
                parts.append(piece)
                print(piece, end="", flush=True)
            if obj.get("done"):
                break
    print()
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="向个人档案库提问")
    parser.add_argument("question", nargs="*", help="问题；不填则进入交互模式")
    parser.add_argument("-k", type=int, default=TOP_K, help="召回片段数")
    parser.add_argument("--sources-only", action="store_true", help="只看召回了什么，不生成")
    args = parser.parse_args()

    index = load(get_embedder())

    def answer(question: str) -> None:
        hits = search(index, question, k=args.k)
        if args.sources_only:
            for hit in hits:
                c = hit.chunk
                print(f"[{hit.score:.4f}] {c.path} › {c.heading} (L{c.tier[-1]})")
                print(f"    {c.text[:120].replace(chr(10), ' ')}...")
            return
        print(f"\n{'-' * 60}\n问: {question}\n{'-' * 60}")
        stream(
            [
                {"role": "system", "content": "你是一个严谨的个人档案问答助手。"},
                {
                    "role": "user",
                    "content": PROMPT.format(
                        context=render(hits), question=question
                    ),
                },
            ]
        )
        print("\n来源：")
        for i, hit in enumerate(hits, 1):
            c = hit.chunk
            print(f"  [{i}] {c.path} › {c.heading}（第 {c.start_line} 行）")

    if args.question:
        answer(" ".join(args.question))
        return 0

    print("交互模式，输入问题，Ctrl-D 退出。")
    for line in sys.stdin:
        q = line.strip()
        if q:
            answer(q)
    return 0


if __name__ == "__main__":
    sys.exit(main())
