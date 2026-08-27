#!/usr/bin/env python3
"""Measure Chinese/Japanese/Korean character leakage in GLM-5.3-Flash Russian output.

GLM-5.3 is a Z.ai model with a Chinese-dominant prior. In Russian generation a CJK
synonym token occasionally outscores the Russian continuation mid-word, e.g.
"разговорное缩短ение". This probe quantifies the rate per sampling configuration so
a mitigation (sampling shape, system prompt, or a logits mask) can be chosen on
evidence rather than vibes.

Colloquial/slang prompts are used on purpose: that is where the observed leak
happened, and where the Russian token distribution is thinnest.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.parse
import urllib.request

CJK = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")

PROMPTS = [
    "йо, чё делаешь",
    "почему все так носятся с этими микросервисами? по-простому, без пафоса",
    "объясни, что значит «раскачиваться перед разговором» и придумай похожее слово",
    "напиши пару строк, как будто ты старый друг, который не любит трёп ни о чём",
    "придумай разговорное сокращение для слова «разогреваешься» и объясни его",
    "переведи на молодёжный сленг: «я устал от бессмысленных совещаний»",
    "поясни за жаргон: что такое «костыль» в коде и откуда пошло",
    "накидай саркастичный ответ на «а можно быстренько один фикс?»",
]

SYSTEM = (
    "Отвечай строго по-русски. Никогда не используй китайские, японские или "
    "корейские иероглифы — ни в ответе, ни в рассуждениях."
)


def ask(url: str, prompt: str, system: str | None, **extra: object) -> tuple[str, str]:
    scheme = urllib.parse.urlsplit(url).scheme
    if scheme not in ("http", "https"):
        raise RuntimeError(f"refusing non-HTTP(S) scheme {scheme!r}")
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body: dict[str, object] = {
        "model": "glm-5.3-flash",
        "messages": messages,
        "max_tokens": 400,
    }
    body.update(extra)
    request = urllib.request.Request(  # noqa: S310 - scheme validated above
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"request failed: {url}: {error}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON from {url}: {error}") from error
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            f"unexpected response shape from {url}: {str(payload)[:200]}"
        ) from error
    return (message.get("content") or ""), (message.get("reasoning") or "")


def run(url: str, label: str, system: str | None, **extra: object) -> dict[str, object]:
    answer_hits = reasoning_hits = answer_chars = dirty_answers = 0
    examples: list[str] = []
    for prompt in PROMPTS:
        content, reasoning = ask(url, prompt, system, **extra)
        hits = CJK.findall(content)
        answer_hits += len(hits)
        reasoning_hits += len(CJK.findall(reasoning))
        answer_chars += len(content)
        if hits:
            dirty_answers += 1
            at = CJK.search(content)
            if at and len(examples) < 3:
                start = max(0, at.start() - 45)
                examples.append(content[start : at.start() + 20].replace("\n", " "))
    per_10k = 10_000 * answer_hits / answer_chars if answer_chars else 0.0
    print(
        f"  {label:34s} answers_with_cjk={dirty_answers}/{len(PROMPTS)}"
        f"  chars={answer_hits:3d} ({per_10k:.2f}/10k)  reasoning={reasoning_hits:3d}"
    )
    for sample in examples:
        print(f"      ...{sample}...")
    return {
        "label": label,
        "answers_with_cjk": dirty_answers,
        "prompts": len(PROMPTS),
        "cjk_chars": answer_hits,
        "cjk_per_10k_chars": round(per_10k, 3),
        "cjk_chars_in_reasoning": reasoning_hits,
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8045/v1/chat/completions")
    parser.add_argument("--output")
    args = parser.parse_args()

    results = [
        run(args.url, "default (t1.0/p0.95)", None),
        run(args.url, "t0.6/p0.9", None, temperature=0.6, top_p=0.9),
        run(args.url, "t0.6/p0.9/top_k40", None, temperature=0.6, top_p=0.9, top_k=40),
        run(args.url, "system-prompt t1.0", SYSTEM),
        run(args.url, "system-prompt t0.6/p0.9", SYSTEM, temperature=0.6, top_p=0.9),
    ]
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(results, handle, ensure_ascii=False, indent=2)
        except OSError as error:
            print(f"could not write {args.output}: {error}")
            return 1
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
