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
docker run --rm --entrypoint sh "$IMAGE" -c '
set -eu
for p in \
  /usr/local/lib/python*/dist-packages/nvidia/cuda_nvrtc/include/nvrtc.h \
  /usr/local/lib/python*/site-packages/nvidia/cuda_nvrtc/include/nvrtc.h
do
  if [ -f "$p" ]; then
    cat "$p"
    exit 0
  fi
done
echo "nvrtc.h not found in nvidia-cuda-nvrtc-cu13 wheel paths" >&2
exit 1
' > "$OUT_DIR/nvrtc.h"

echo "wrote $OUT_DIR/nvrtc.h from nvidia-cuda-nvrtc-cu13 inside $IMAGE"
