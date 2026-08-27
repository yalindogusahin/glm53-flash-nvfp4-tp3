#!/usr/bin/env python3
"""Build a GB10 overlay for vLLM's kpool sparse-attention indexer.

Why
---
GLM-5.3-Flash's DeepSeek-sparse layers pick `index_topk` keys through
`torch.ops._C.persistent_topk`. That kernel is only launched when
`select_k = index_topk // index_kpool` is one of (512, 1024, 2048), which is
exactly GLM-5.3's case (2048 // 4 == 512). On a GB10 the launcher refuses
above roughly 25k tokens of context:

    persistent_topk would oversubscribe and the FilteredTopK fallback
    requires >=128KB smem per block (have 101376).
    total_ctas=62 > num_sms*occupancy=48 (TopK=512, ctas_per_group=62)

101,376 B is the GB10 opt-in shared-memory ceiling, so the built-in
FilteredTopK fallback can never fit, and the request kills the engine.

The same function already has a generic CUDA path,
`torch.ops._C.top_k_per_row_decode`, used whenever `select_k` is an
unsupported width. It has no CTA/smem precondition. This overlay keeps the
fast kernel for short contexts and degrades to the generic one instead of
dying when the launcher refuses.

Usage
-----
    glm53-indexer-topk-fallback.py --image glm53:sm121-v8 --out ~/src/glm53-overlay

Writes `<out>/vllm/model_executor/layers/sparse_attn_indexer_kpool.py`.
Refuses to patch when the extracted source no longer matches, so a base-image
change is loud rather than silent.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

REL_PATH = "vllm/model_executor/layers/sparse_attn_indexer_kpool.py"
IN_IMAGE = f"/usr/local/lib/python3.12/dist-packages/{REL_PATH}"

CALL = """            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_dst,
                topk_workspace,
                select_k,
                attn_metadata_narrowed.max_seq_len,
            )
"""

REPLACEMENT = """            try:
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    topk_dst,
                    topk_workspace,
                    select_k,
                    attn_metadata_narrowed.max_seq_len,
                )
            except RuntimeError as persistent_topk_error:
                # GB10 (sm121) opt-in smem is 101,376 B, below the 128 KB the
                # built-in FilteredTopK fallback needs, so the launcher refuses
                # once ctas_per_group exceeds num_sms*occupancy (~25k tokens of
                # context). Use the generic per-row kernel instead of letting
                # the engine die.
                if "persistent_topk" not in str(persistent_topk_error):
                    raise
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )
"""


def extract(image: str, dest: Path) -> str:
    docker = shutil.which("docker")
    if docker is None:
        raise SystemExit("docker not found in PATH")
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(  # noqa: S603 - fixed argv, docker resolved from PATH
        [docker, "run", "--rm", "--entrypoint", "cat", image, IN_IMAGE],
        capture_output=True,
        check=True,
    )
    return proc.stdout.decode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="glm53:sm121-v8")
    parser.add_argument("--out", type=Path, default=Path.home() / "src/glm53-overlay")
    args = parser.parse_args()

    target = args.out / REL_PATH
    source = extract(args.image, target)

    count = source.count(CALL)
    if count != 1:
        raise SystemExit(
            f"expected exactly 1 persistent_topk call site, found {count}; "
            "base image changed, refusing to patch"
        )
    for needed in ("torch.ops._C.top_k_per_row_decode(", "next_n", "num_rows"):
        if needed not in source:
            raise SystemExit(f"fallback prerequisite {needed!r} missing; refusing")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.replace(CALL, REPLACEMENT))
    print(f"wrote {target} ({target.stat().st_size} bytes, 1 call site guarded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
