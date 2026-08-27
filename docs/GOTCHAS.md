# Gotchas

Ordered by how much time each one costs you.

## 1. Requests past ~25K tokens kill the engine (GB10 only)

```text
RuntimeError: launch_persistent_topk, csrc/libtorch_stable/topk.cu:138,
persistent_topk would oversubscribe and the FilteredTopK fallback requires
>=128KB smem per block (have 101376).
total_ctas=62 > num_sms*occupancy=48 (TopK=512, ctas_per_group=62)
```

Not the request — the whole engine dies and all three ranks must be relaunched.

GLM-5.3's DeepSeek-sparse layers select `index_topk // index_kpool = 2048 // 4 = 512` keys,
and `select_k in (512, 1024, 2048)` is exactly what routes vLLM into
`torch.ops._C.persistent_topk`. Past ~25K tokens that kernel wants more CTAs than a GB10 has
SM occupancy for, and its built-in FilteredTopK fallback needs ≥128 KiB shared memory while
**GB10's opt-in smem ceiling is 101,376 B** — the fallback can never fit.

Fix: `scripts/build-indexer-overlay.py` guards the fast call with `except RuntimeError` and
degrades to `torch.ops._C.top_k_per_row_decode`, the generic path already present in the same
function for unsupported `select_k` widths. Short contexts keep the fast kernel. Mount the
result over the image (`serve.sh` does this when the file exists).

Upstream repos advertise 1M context; that is KV-pool arithmetic, not a long prefill actually
executed.

## 2. `--reasoning-parser glm45` silently merges reasoning into the answer

The model card says `glm45`. Use **`glm47`**.

`chat_template.jinja` ends the prompt with `<|assistant|><think>` — the reasoning block is
**prefilled**, so the model's output starts *inside* it and only ever emits the closing tag.
Raw output via `/v1/completions`:

```text
17 × 23 = 391\n\nLet me verify: ... Correct.</think>17 × 23 = 391.
```

`glm45` waits for an opening `<think>` in the output, never sees one, and concatenates
everything into `content`. `glm47`'s parser config starts in `ParserState.REASONING`.

## 3. `enable_thinking:false` does not disable thinking — it breaks the split

The jinja ignores `enable_thinking` entirely (there is no non-thinking path). All the flag
does is tell the parser to start in `CONTENT`, which re-merges reasoning into `content`.

Server default must be `--default-chat-template-kwargs '{"enable_thinking":true}'`.

Also: this vLLM build returns **`reasoning`** / `reasoning_details`, not `reasoning_content`.
Clients and harnesses that only read `reasoning_content` will report empty reasoning.

## 4. Reasoning effort: low / high / max only

```jinja
set effective_reasoning_effort = reasoning_effort
  if reasoning_effort in ['low','high'] else 'max'
```

No medium, no xhigh, no off; default is **max**. Steer with
`chat_template_kwargs: {"reasoning_effort": "low"}`. Budget `max_tokens` accordingly — 200
tokens can be fully consumed by reasoning, leaving an empty `content`.

`clear_thinking: true` strips `<think>` from previous turns (keeps `<think></think>`), which
is worth it in long agent loops.

## 5. On-disk MoE width must be 2112, not 2049

Two independent consumers must agree:

- the target model reads `--hf-overrides` (`moe_intermediate_size: 2112`),
- `SpeculativeConfig` for native MTP reads the **checkpoint's** `config.json`.

Upstream's pad script writes `ceil(2048/3)*3 = 2049`, while its own overlay forces 2112 at
TP=3. A 2049 on-disk value gives the MTP draft 683-wide shards against a 704-wide target, and
683 is not a multiple of the NVFP4 16-value block scale. `scripts/pad-config.py` here writes
66 heads / **2112** and keeps a `config.json.orig`.

## 6. The NVFP4 repo has no chat template inside `tokenizer_config.json`

`transformers ≥ 4.44` refuses to invent one:

```text
As of transformers v4.44, default chat template is no longer allowed...
```

`chat_template.jinja` exists as a separate file in the HF repo, and weight-oriented
downloaders (e.g. anything filtering `*.safetensors` + `*.json`) skip it. Fetch it explicitly
and pass `--chat-template`. `scripts/fetch-weights.sh` does both.

## 7. PP=3 is not an alternative to patched TP=3

Pipeline parallelism looks attractive (45 layers ÷ 3, no divisibility problem, ~60 GiB/rank),
but non-first pipeline ranks never receive the mHC `post`/`comb` tensors and
`mhc_pre_tilelang` asserts during the profile forward. For three nodes, patched TP=3 is the
only working geometry.

## 8. RDMA rendezvous cannot live on the fabric (triangle wiring)

Three Sparks with two ports each usually form a **triangle of pairwise links**, not one
subnet: rank 1 sees rank 0 on one /24, rank 2 sees it on another. No single `--master-addr`
on the fabric is reachable from both workers, so TCPStore/Gloo must sit on a shared L2 (1 GbE
management is fine). Only the rendezvous goes there — measured over 600 decoded tokens:
**0.5 MB on 1 GbE vs 2,342 MB on RDMA**.

`TRANSPORT=ib` works on this topology with `NCCL_IB_SUBNET_AWARE_ROUTING=1` and all logical
HCAs listed, without pinning `NCCL_IB_GID_INDEX`. The upstream `nccl-mesh` plugin
(`TRANSPORT=mesh`) remains available if your wiring defeats subnet-aware routing.

## 9. Smaller traps

- **`min_p` is rejected** (HTTP 400) by this build. Use `top_p` / `top_k`.
- **`docker` group**: a node provisioned later than the others may not have it. And `ssh`
  ControlMaster reuse will keep serving you a *stale* session without the new group —
  use `-o ControlPath=none` after `usermod -aG`.
- **`docker save | docker load`** produces a different image ID on the receiving node (the
  manifest list is flattened). Compare content instead, e.g. hash `platforms/cuda.py` and
  print `flashinfer.__version__` inside the container.
- **Drop page cache before each launch** (`sync; echo 3 > /proc/sys/vm/drop_caches`). GB10's
  allocator competes with page cache for the KV slab.
- **`0x51` kernel events are expected** during FlashInfer autotuning. Gate on `Xid` instead.
- **`--max-num-batched-tokens 8192`** keeps chunked prefill predictable; the default is fine
  for short prompts but this bounds indexer work per step.
