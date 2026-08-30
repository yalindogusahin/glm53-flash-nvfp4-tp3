# Provenance of this overlay

Three distinct origins live in this directory. Keep them straight before editing.

## Ported verbatim from upstream vLLM (Apache-2.0)

`qwen3_dflash2.py` and `dflash2/speculator.py` carry
`SPDX-License-Identifier: Apache-2.0` and
`SPDX-FileCopyrightText: Copyright contributors to the vLLM project`.

They come from vLLM PR #52816, "[Spec Decode] DFlash2: local convolution +
candidate selector", **merged upstream on 2026-08-21**. Both files exist in vLLM
`main` today (`vllm/model_executor/models/qwen3_dflash2.py`,
`vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py`).

Do NOT replace these copies with the versions from `main`. The GLM-5.3-Flash
image this repo builds on pins vLLM `g487ecf187` (~2026-08-15), six days before
that merge, and the ported copies carry `SM121-PORT:` comments marking every
deviation needed on that older tree - for example `gumbel_noised_argmax` is
defined locally because the image's `gumbel.py` predates the refactor that
factored it out. The `main` versions will not import here.

Once a GLM-5-Next image ships with post-merge vLLM, both files can be dropped.

## Original work by @yalindogusahin, from PR #1 against this repo

- `patch_glm_aux_capture.py` - EAGLE3 aux-hidden-state capture in the GLM model
- `patch_glm5_drafter_group.py` - GLM-5-Next KV grouping for the drafter's
  `SlidingWindowSpec` layers
- `patch_kv_page_lcm2.py`, `patch_registry_and_select.py` - registration and KV
  page LCM glue for the pre-merge tree
- `sim_glm5_drafter.py`, `NOTES.md`, `GLUE-NOTES.md`, `patch_v8_fp8.py`
- `../../scripts/pad-dflash2-drafter.py` - the 48 q / 12 kv TP=3 drafter pad
- `../../scripts/extract-nvrtc-header.sh`

None of this is available upstream: GLM-5-Next is not in vLLM `main`
(`vllm/model_executor/models/glm5next.py` returns 404 there, and `main`'s
`kv_cache_utils.py` has no `glm5_next` grouping at all). It lives only in the
`glm53-flash` image tree, so these patches had to be written against it.

Source: https://github.com/jetnet/glm53-flash-nvfp4-tp3/pull/1

## Changed locally

`../Dockerfile.glm53-sm121-v12-dflash2` takes `BASE_IMAGE` as a build ARG and
defaults to the `glm53:sm121-v8` this repo builds, instead of the contributor's
third-party GHCR base.

Deliberately NOT taken from PR #1: the default-checkpoint switch to
`RedHatAI/GLM-5.3-Flash-NVFP4` (a `compressed-tensors` mixed-precision build with
**quantized activations**, unlike the weight-only A16 ModelOpt checkpoint this
repo runs) - that is a separate decision needing its own measurement.
