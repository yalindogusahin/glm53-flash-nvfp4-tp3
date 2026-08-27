#!/usr/bin/env python3
"""Pad GLM-5.3-Flash's config.json so TP=3 is legal. Idempotent.

`--hf-overrides` covers the target model, but native MTP's SpeculativeConfig reads the
checkpoint file, so the file itself must be padded: 64 heads -> 66, MoE intermediate
2048 -> 2112.

2112, not ceil(2048/3)*3 = 2049: the overlay forces 2112 at TP=3 (it is a multiple of 64,
so NVFP4's 16-value block scales and Marlin's tiling stay aligned). A 2049 on-disk value
gives the MTP draft 683-wide shards against the target's 704 and fails to load.

Backs up config.json.orig once.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

GEMM_FRIENDLY_MOE = {(2048, 3): 2112}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--tp", type=int, default=3)
    args = parser.parse_args()

    try:
        data = json.loads(args.config.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read {args.config}: {error}") from error

    text = data.get("text_config")
    if not isinstance(text, dict):
        raise SystemExit(f"{args.config}: no text_config")

    tp = args.tp
    ceil_tp = lambda n: (n + tp - 1) // tp * tp  # noqa: E731
    heads = ceil_tp(int(text.get("num_attention_heads", 64)))
    moe_in = int(text.get("moe_intermediate_size", 2048))
    moe = GEMM_FRIENDLY_MOE.get((moe_in, tp), ceil_tp(moe_in))

    before = json.dumps(data, sort_keys=True)
    for key in ("num_attention_heads", "num_key_value_heads", "linear_num_heads"):
        text[key] = heads
    text["moe_intermediate_size"] = moe
    linear = text.get("linear_attn_config")
    if isinstance(linear, dict) and "num_heads" in linear:
        linear["num_heads"] = heads
    # vocab_size stays 154880: the embedding pads in-module (padding_size = lcm(64, tp)).

    if json.dumps(data, sort_keys=True) == before:
        print(f"already padded: heads={heads} moe_intermediate_size={moe}")
        return 0

    backup = args.config.with_suffix(args.config.suffix + ".orig")
    try:
        if not backup.exists():
            shutil.copy2(args.config, backup)
            print(f"backup -> {backup}")
        args.config.write_text(json.dumps(data, indent=2) + "\n")
    except OSError as error:
        raise SystemExit(f"cannot write {args.config}: {error}") from error
    print(f"padded: heads={heads} moe_intermediate_size={moe} (tp={tp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
