#!/usr/bin/env python3
"""Needle-in-a-haystack retrieval at a chosen prompt length.

Use --thinking on for GLM-5.3: its template prefills <think>, so the answer only appears
after the model closes the block - give --max-tokens room and score by containment.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def post(
    url: str, payload: dict[str, object], api_key: str, timeout: int
) -> dict[str, Any]:
    # Only ever talk HTTP(S) to a cluster endpoint: refuse file:/ftp:/custom schemes
    # so a mistyped --base-url cannot turn this into a local file read.
    scheme = urllib.parse.urlsplit(url).scheme
    if scheme not in ("http", "https"):
        raise RuntimeError(f"refusing non-HTTP(S) URL scheme {scheme!r}: {url}")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(  # noqa: S310 - scheme validated above
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - scheme validated above
            request, timeout=timeout
        ) as response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"request failed: {url}: {error}") from error
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON from {url}") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"unexpected response from {url}: {type(result).__name__}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8045/v1")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--target-tokens", type=int, default=238_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--thinking",
        choices=("off", "on"),
        default="off",
        help=(
            "on: send enable_thinking=true and score with substring containment. "
            "Required for reasoning-always models such as GLM-5.3-Flash, whose "
            "template prefills <think> and needs headroom to close it."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "high"),
        default=None,
        help="GLM-5.3 reasoning_effort; omitted means the model default (max).",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path.home() / ".config/glm53/api-key",
    )
    args = parser.parse_args()

    api_key = (
        args.api_key_file.read_text().strip() if args.api_key_file.exists() else ""
    )
    tokenize_url = args.base_url.removesuffix("/v1") + "/tokenize"
    completion_url = args.base_url.rstrip("/") + "/chat/completions"
    secret = f"NEEDLE-{uuid.uuid4().hex.upper()}"
    unit = "\nArchival filler record: ordinary context with no secret value."
    prefix = (
        "Read the complete archival record. Remember the exact secret in its middle. "
        "At the end, reply with only that secret and no punctuation.\n"
    )
    marker = f"\nSECRET RECORD: {secret}\n"
    suffix = "\nEND OF ARCHIVE. What is the exact secret? Reply with only the secret."

    def count(text: str) -> int:
        result = post(
            tokenize_url,
            {"model": args.model, "prompt": text},
            api_key,
            timeout=300,
        )
        value = result.get("count")
        if not isinstance(value, int):
            raise RuntimeError(f"invalid token count: {value!r}")
        return value

    low, high = 0, args.target_tokens
    best_text = prefix + marker + suffix
    best_count = count(best_text)
    while low <= high:
        middle = (low + high) // 2
        text = prefix + unit * middle + marker + unit * middle + suffix
        tokens = count(text)
        if tokens < args.target_tokens:
            best_text, best_count = text, tokens
            low = middle + 1
        elif tokens > args.target_tokens + 8:
            high = middle - 1
        else:
            best_text, best_count = text, tokens
            break

    while best_count < args.target_tokens:
        missing = args.target_tokens - best_count
        best_text = best_text.removesuffix(suffix) + (" filler" * missing) + suffix
        best_count = count(best_text)

    print(f"built_prompt_tokens={best_count} target={args.target_tokens}", flush=True)
    payload: dict[str, object] = {
        "model": args.model,
        "messages": [{"role": "user", "content": best_text}],
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "seed": 42,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": args.thinking == "on"},
    }
    if args.reasoning_effort:
        template_kwargs = payload["chat_template_kwargs"]
        if isinstance(template_kwargs, dict):
            template_kwargs["reasoning_effort"] = args.reasoning_effort
    started = time.monotonic()
    result = post(completion_url, payload, api_key, timeout=1800)
    elapsed = time.monotonic() - started
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = str(message.get("content") or "").strip()
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    summary = {
        "target_tokens": args.target_tokens,
        "tokenized_prompt_tokens": best_count,
        "usage": result.get("usage"),
        "elapsed_s": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "reasoning_chars": len(reasoning),
        "secret": secret,
        "exact_match": content == secret,
        "contains_secret": secret in content,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    passed = (
        summary["contains_secret"] if args.thinking == "on" else summary["exact_match"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
