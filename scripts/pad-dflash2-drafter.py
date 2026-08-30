#!/usr/bin/env python3
"""
Pad the DFlash2 drafter's attention heads so it can shard at TP=3.

The drafter (incoai/GLM-5.3-Flash-DFlash2) is 32 attention heads / 8 KV heads.
Neither divides by 3, and DFlashQwen3Attention asserts both:

    assert self.total_num_heads % tp_size == 0
    assert self.total_num_kv_heads % tp_size == 0      # when kv >= tp

draft_tensor_parallel_size=1 does NOT help: load_dflash_model() builds the draft
under the *target's* parallel config and never applies draft_parallel_config, so
the drafter always shards at target TP. Upstream only ran DFlash2 at TP=2 and
TP=4, where 32 and 8 both divide.

Pad to 48 Q / 12 KV -- NOT the smaller 36/9, and the reason is KV geometry, not
head divisibility. Both pads shard at TP=3 and both preserve the GQA ratio of 4,
but the KV cache page math differs sharply:

    pad     kv/rank   B/token   exact-fit block   vs MLA block 4608   mode
    36/9          3       768              3072   4608 % 3072 != 0    STANDALONE
    48/12         4      1024              2304   4608 % 2304 == 0    EXACT FIT

`patch_glm5_drafter_group.py` can only slot-share the drafter onto the MLA
tensors when its rescaled page exactly fills the MLA page AND the two block
sizes divide one another. At 3 kv heads/rank that mutual-divisibility gate
fails, the drafter falls to the standalone path keeping its block_size=16, and
one 512K request demands ~20 GiB instead of ~3.2 GiB -- it will not boot against
a 10.73 GiB pin. Upstream also records that the standalone path "had never run
on real hardware" and is their prime suspect for a chunked-prefill kill.

At 4 kv heads/rank the drafter is per-rank geometrically IDENTICAL to upstream's
working TP=2 lane (exact-fit block 2304, sim GEOMETRY A), which is the only
configuration proven on hardware.

    q_proj  [32*128, H] -> [48*128, H]  16 appended head blocks, zeroed
    k_proj  [ 8*128, H] -> [12*128, H]   4 appended head blocks, zeroed
    v_proj  [ 8*128, H] -> [12*128, H]   4 appended head blocks, zeroed
    o_proj  [H, 32*128] -> [H, 48*128]  16 appended column blocks, zeroed

q head i attends kv head i//4 throughout, so every real head (i < 32) keeps its
original kv head. The 16 padded q heads read padded kv heads 8..11 and
contribute exactly zero: q=0 gives a uniform softmax over v=0, and their o_proj
columns are zero regardless. q_norm/k_norm are per-head-dim (shape [128]) and
apply identically to every head, so they need no change.

Everything else -- MLP, the DFlash convolutions, the candidate selector and its
codebooks -- is a function of hidden_size only and is copied verbatim.

Usage:
    pad-dflash2-drafter.py SRC_DIR DST_DIR [--tp 3] [--verify-only]

Idempotent in the sense that it always writes DST_DIR from scratch; it refuses
to run if SRC_DIR is already padded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

HEAD_DIM_KEY = "head_dim"
Q_HEADS = 32
KV_HEADS = 8
PAD_Q_HEADS = 48
PAD_KV_HEADS = 12


def pad_rows(w: torch.Tensor, n_src_heads: int, n_dst_heads: int, head_dim: int) -> torch.Tensor:
    """Append zeroed head blocks along dim 0 (q/k/v projections)."""
    assert w.shape[0] == n_src_heads * head_dim, (w.shape, n_src_heads, head_dim)
    pad = torch.zeros(
        (n_dst_heads - n_src_heads) * head_dim, w.shape[1], dtype=w.dtype
    )
    return torch.cat([w, pad], dim=0)


def pad_cols(w: torch.Tensor, n_src_heads: int, n_dst_heads: int, head_dim: int) -> torch.Tensor:
    """Append zeroed head blocks along dim 1 (o_proj input)."""
    assert w.shape[1] == n_src_heads * head_dim, (w.shape, n_src_heads, head_dim)
    pad = torch.zeros(
        w.shape[0], (n_dst_heads - n_src_heads) * head_dim, dtype=w.dtype
    )
    return torch.cat([w, pad], dim=1)


def sdpa_ref(q, k, v, n_heads, n_kv_heads, head_dim):
    """Reference GQA attention for one token block; [T, n_heads*head_dim] -> [T, ...]."""
    t = q.shape[0]
    q = q.view(t, n_heads, head_dim).transpose(0, 1)            # [Hq, T, D]
    k = k.view(t, n_kv_heads, head_dim).transpose(0, 1)         # [Hkv, T, D]
    v = v.view(t, n_kv_heads, head_dim).transpose(0, 1)
    rep = n_heads // n_kv_heads
    k = k.repeat_interleave(rep, dim=0)
    v = v.repeat_interleave(rep, dim=0)
    scores = (q.float() @ k.float().transpose(-1, -2)) * head_dim**-0.5
    out = torch.softmax(scores, dim=-1) @ v.float()             # [Hq, T, D]
    return out.transpose(0, 1).reshape(t, n_heads * head_dim)


def verify(src: dict, dst: dict, head_dim: int, layer: str = "layers.0") -> None:
    """The padded layer must produce bit-comparable output to the original.

    Runs one attention layer both ways on the same random hidden states and
    compares the o_proj output. The padded heads must contribute nothing.
    """
    torch.manual_seed(0)
    hidden = src[f"{layer}.self_attn.q_proj.weight"].shape[1]
    x = torch.randn(16, hidden, dtype=torch.float32)

    def run(w, nq, nkv):
        q = x @ w[f"{layer}.self_attn.q_proj.weight"].float().T
        k = x @ w[f"{layer}.self_attn.k_proj.weight"].float().T
        v = x @ w[f"{layer}.self_attn.v_proj.weight"].float().T
        a = sdpa_ref(q, k, v, nq, nkv, head_dim)
        return a @ w[f"{layer}.self_attn.o_proj.weight"].float().T

    ref = run(src, Q_HEADS, KV_HEADS)
    got = run(dst, PAD_Q_HEADS, PAD_KV_HEADS)
    err = (ref - got).abs().max().item()
    scale = ref.abs().max().item()
    print(f"[verify] {layer}: max|ref-padded| = {err:.3e}  (output scale {scale:.3e})")
    if err > 1e-4 * max(scale, 1.0):
        sys.exit(f"VERIFY FAILED: padded attention output differs by {err}")
    print("[verify] OK -- padded heads contribute exactly nothing.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--tp", type=int, default=3)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.src, "config.json")))
    head_dim = cfg[HEAD_DIM_KEY]
    if cfg["num_attention_heads"] != Q_HEADS or cfg["num_key_value_heads"] != KV_HEADS:
        sys.exit(
            f"expected an unpadded {Q_HEADS}/{KV_HEADS} drafter, got "
            f"{cfg['num_attention_heads']}/{cfg['num_key_value_heads']}"
        )
    for n in (PAD_Q_HEADS, PAD_KV_HEADS):
        if n % args.tp:
            sys.exit(f"pad target {n} does not divide TP={args.tp}")
    assert PAD_Q_HEADS // PAD_KV_HEADS == Q_HEADS // KV_HEADS, "GQA ratio must survive"

    src_w = load_file(os.path.join(args.src, "model.safetensors"))
    dst_w = {}
    touched = 0
    for k, w in src_w.items():
        if k.endswith("self_attn.q_proj.weight"):
            dst_w[k] = pad_rows(w, Q_HEADS, PAD_Q_HEADS, head_dim)
        elif k.endswith("self_attn.k_proj.weight") or k.endswith("self_attn.v_proj.weight"):
            dst_w[k] = pad_rows(w, KV_HEADS, PAD_KV_HEADS, head_dim)
        elif k.endswith("self_attn.o_proj.weight"):
            dst_w[k] = pad_cols(w, Q_HEADS, PAD_Q_HEADS, head_dim)
        else:
            dst_w[k] = w
            continue
        touched += 1
    print(f"[pad] {touched} attention tensors padded, {len(src_w) - touched} copied verbatim")

    verify(src_w, dst_w, head_dim)
    if args.verify_only:
        return 0

    os.makedirs(args.dst, exist_ok=True)
    for extra in os.listdir(args.src):
        if extra not in ("model.safetensors", "config.json"):
            s = os.path.join(args.src, extra)
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(args.dst, extra))

    cfg["num_attention_heads"] = PAD_Q_HEADS
    cfg["num_key_value_heads"] = PAD_KV_HEADS
    cfg["_tp3_head_pad"] = {
        "orig_num_attention_heads": Q_HEADS,
        "orig_num_key_value_heads": KV_HEADS,
        "note": "zero-padded for TP=3; padded heads contribute nothing",
    }
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    save_file(dst_w, os.path.join(args.dst, "model.safetensors"), metadata={"format": "pt"})
    print(f"[pad] wrote {args.dst}: {PAD_Q_HEADS} q / {PAD_KV_HEADS} kv heads, TP={args.tp} OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
