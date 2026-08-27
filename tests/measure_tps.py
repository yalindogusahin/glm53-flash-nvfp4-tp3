#!/usr/bin/env python3
"""Streaming TTFT / decode-tok/s probe. Thinking off, temperature 0.

From FlyCockpit/GLM-5.3-Flash-3x-DGX-Sparks (MIT), port default adjusted.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8045/v1")
    p.add_argument("--model", default="glm-5.3-flash")
    p.add_argument("--prompt", default="Say hello and name yourself in one sentence.")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--timeout", type=int, default=300)
    args = p.parse_args()

    payload = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()

    for run in range(1, args.runs + 1):
        req = urllib.request.Request(
            args.url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        first_token = None
        completion_tokens = None
        finish_reason = None
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if first_token is None and (
                        delta.get("content")
                        or delta.get("reasoning_content")
                        or delta.get("reasoning")
                    ):
                        first_token = time.perf_counter()
                    finish_reason = choices[0].get("finish_reason") or finish_reason
                usage = event.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens")
        end = time.perf_counter()
        ttft = (first_token - start) if first_token is not None else float("nan")
        decode_s = max(end - (first_token or start), 1e-9)
        decode_tokens = max((completion_tokens or 0) - 1, 0)
        print(
            json.dumps(
                {
                    "run": run,
                    "ttft_s": round(ttft, 4),
                    "total_s": round(end - start, 4),
                    "completion_tokens": completion_tokens,
                    "decode_tok_s": round(decode_tokens / decode_s, 3),
                    "finish_reason": finish_reason,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
