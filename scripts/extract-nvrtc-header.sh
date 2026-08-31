#!/usr/bin/env bash
#
# Extract nvrtc.h from the NVIDIA CUDA NVRTC Python wheel installed in the
# vLLM image. FlashInfer JIT needs this one header, but the base image does
# not place it under /usr/local/cuda/include.
#
#   scripts/extract-nvrtc-header.sh [image] [out-dir]
#
# Example:
#   scripts/extract-nvrtc-header.sh glm53:sm121-v12-dflash2 ~/src/glm53-overlay
#
set -euo pipefail

IMAGE="${1:-glm53:sm121-v12-dflash2}"
OUT_DIR="${2:-$HOME/src/glm53-overlay}"

mkdir -p "$OUT_DIR"

# Write to a temp file first. Redirecting straight at the destination truncates it
# to zero bytes when the lookup inside the image fails, and a zero-byte nvrtc.h is
# worse than a missing one: serve.sh happily bind-mounts it and FlashInfer JIT then
# fails at warmup with a confusing compile error instead of a clear "no header".
TMP=$(mktemp "${TMPDIR:-/tmp}/nvrtc.h.XXXXXX")
trap 'rm -f "$TMP"' EXIT

# Wheel layouts differ between base images: older ones ship nvidia/cuda_nvrtc/,
# current cu13 wheels consolidate everything under nvidia/cu13/.
docker run --rm --entrypoint sh "$IMAGE" -c '
set -eu
for p in \
  /usr/local/lib/python*/dist-packages/nvidia/cu*/include/nvrtc.h \
  /usr/local/lib/python*/site-packages/nvidia/cu*/include/nvrtc.h \
  /usr/local/lib/python*/dist-packages/nvidia/cuda_nvrtc/include/nvrtc.h \
  /usr/local/lib/python*/site-packages/nvidia/cuda_nvrtc/include/nvrtc.h
do
  if [ -f "$p" ]; then
    cat "$p"
    exit 0
  fi
done
echo "nvrtc.h not found under any known nvidia wheel include path" >&2
exit 1
' > "$TMP"

test -s "$TMP" || { echo "extraction produced an empty nvrtc.h - refusing to install it" >&2; exit 1; }
grep -q "nvrtcResult" "$TMP" || { echo "extracted file does not look like nvrtc.h" >&2; exit 1; }

cp "$TMP" "$OUT_DIR/nvrtc.h"
echo "wrote $OUT_DIR/nvrtc.h ($(wc -c <"$OUT_DIR/nvrtc.h") bytes) from $IMAGE"
