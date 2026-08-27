# Measurements

3× DGX Spark (GB10 / SM121, ~121 GiB unified memory each), driver 580.173.02,
kernel 6.17.0-1026-nvidia, Docker 29.2.1. Image `glm53:sm121-v8` rebuilt from the public
day-0 base `vllm/vllm-openai:glm53-flash-arm64-cu130` (vLLM `0.1.dev20051`,
FlashInfer `0.6.18.dev20260819`, NCCL 2.30.7).

Checkpoint: `LibertAIDAI/GLM-5.3-Flash-NVFP4`, 182 GiB / 120 shards, **one copy**, read by
ranks 1–2 over NFSv4.2 on the RDMA fabric. GPU clock capped at 1500 MHz unless stated.

Date: 2026-08-27.

## Throughput

Prompt: "Write a Python function that merges two sorted lists, with a short docstring.",
300 tokens, temperature 0, 3 runs.

| Profile | Weights/rank | KV pin | KV pool | Conc. @ len | TTFT | Decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no spec, 262K | 62.14 GiB | none | 4,718,592 | 18.00× | 0.20 s | — |
| no spec, 262K | 62.14 GiB | 4.38 GiB | 675,219 | 2.58× | 0.16 s | 18.4 tok/s |
| MTP-4, 262K | 63.64 GiB | 4.38 GiB | 534,635 | 2.04× | 0.18 s | 32.0 tok/s |
| **MTP-4, 512K** | 63.79 GiB | 10.73 GiB | 1,505,849 | 2.87× | 0.26 s | **35.2 tok/s** |

MTP-4 is **1.9×** over no speculation. Run-to-run spread is wide (32.1–39.0 tok/s) because
draft acceptance depends on the text.

Startup: 462–479 s to load 120 shards, +170 s engine warmup.

## Long context

Exact-string needle retrieval, `tests/long_context_needle.py --thinking on`:

| Prompt tokens | Result | Wall clock | Prefill |
| ---: | --- | ---: | ---: |
| 16,013 | PASS | 10.8 s | — |
| 64,013 | PASS | 44.5 s | 1,440 tok/s |
| 131,013 | PASS | 73.8 s | 1,775 tok/s |
| 238,013 | PASS | 126.2 s | 1,886 tok/s |
| **471,813** | **PASS** | 274.6 s | 1,718 tok/s |

All of these require the indexer overlay. Without it, anything past ~25K tokens takes the
engine down (see GOTCHAS #1).

## GPU clock

| Clock | Decode median | TTFT | 64K prefill | Temp |
| --- | ---: | ---: | ---: | ---: |
| 1500 MHz | 35.23 tok/s | 0.268 s | 44.5 s | 53–57 °C |
| 2100 MHz | 36.79 tok/s | 0.259 s | **35.3 s** | 53–58 °C |
| delta | +4.4% | −3% | **−26%** | +1 °C |

Decode is fabric-bound: +40% clock buys +4%. Prefill is GEMM-bound and gains 26%. Raise the
clock if your workload is prompt-heavy.

## Fabric split

Interface counters on rank 0 across 600 decoded tokens:

| Path | Bytes |
| --- | ---: |
| 1 GbE management (rendezvous, Gloo, control RPC) | 0.5 MB |
| RDMA fabric (tensor traffic) | 2,342 MB |

Ratio 4,335×. `TRANSPORT=ib` with `NCCL_IB_SUBNET_AWARE_ROUTING=1` over four logical HCAs
worked on a pairwise-triangle wiring without the upstream mesh plugin and without pinning
`NCCL_IB_GID_INDEX`.

## Behaviour

- Tool calling (`--tool-call-parser glm47`): `finish_reason=tool_calls`,
  `get_weather {"city": "Berlin"}`.
- Reasoning split with `--reasoning-parser glm47` and `enable_thinking:true`: a plain "yo"
  returns 618 chars of `reasoning` and `content = "Hey! What's up?"`.
- CJK leakage in Russian output: **0 characters in 40 answers** (8 colloquial prompts × 5
  sampling configurations, including default `t1.0/p0.95`), `tests/cjk_leak_probe.py`. A
  single real-world occurrence was reported but is not reproducible at this rate.
- Xid 0 on all three ranks throughout, including the 471K run. `0x51` events appear during
  FlashInfer autotuning and are benign — gate on Xid.

## Not tested

- `--enable-expert-parallel` (288 experts ÷ 3 = 96/rank) booted upstream with no weight
  saving; not measured here.
- Official `zai-org/GLM-5.3-Flash` FP8 (~328 GB → ~102 GiB/rank). With NVFP4 already at
  62–64 GiB/rank and nodes at 99–115 GiB used, FP8 leaves 8–17 GiB for engine, activations
  and KV, plus unproven FP8 MoE kernels on sm121.
- `local-inference-lab/GLM-5.3-Flash-NVFP4` (185.7 GiB): quantizes **activations** to FP4
  (group 16, static, with a real `amax_checkpoint.json` calibration) and keeps the MTP
  layer-45 experts in MXFP8, unlike LibertAI's weight-only NVFP4-A16. Different speed and
  quality profile; also unclear whether `--moe-backend marlin` accepts A4 on GB10.
