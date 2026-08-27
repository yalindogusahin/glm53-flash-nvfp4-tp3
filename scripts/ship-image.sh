#!/usr/bin/env bash
#
# Copy a locally built image to peers over the fastest link they answer SSH on.
#
#   ship-image.sh glm53:sm121-v8 <peer> [peer...]
#
# Note: docker save|load flattens the manifest list, so the receiving node reports a
# different image ID. Compare content, not IDs - this script does that for you.
set -euo pipefail

IMAGE="${1:?usage: ship-image.sh <image> <peer> [peer...]}"
shift
[ "$#" -ge 1 ] || { echo "no peers given" >&2; exit 2; }

docker image inspect "$IMAGE" >/dev/null

fingerprint() {
	# $1 = "" for local, otherwise an ssh target
	local runner=(docker)
	[ -n "$1" ] && runner=(ssh -o BatchMode=yes -o ControlPath=none "$1" docker)
	"${runner[@]}" run --rm --entrypoint python3 "$IMAGE" -c '
import hashlib, pathlib, flashinfer
b = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
h = lambda p: hashlib.sha256((b / p).read_bytes()).hexdigest()[:12]
print(flashinfer.__version__,
      h("platforms/cuda.py"),
      h("v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"))' 2>/dev/null | tail -1
}

LOCAL="$(fingerprint "")"
echo "local  $IMAGE: $LOCAL"

for peer in "$@"; do
	# ControlPath=none: a reused ssh master can predate `usermod -aG docker`.
	if ssh -o BatchMode=yes -o ControlPath=none "$peer" docker image inspect "$IMAGE" >/dev/null 2>&1; then
		echo "$peer: already present, skipping transfer"
	else
		echo "$peer: sending..."
		docker save "$IMAGE" | ssh -o BatchMode=yes -o ControlPath=none "$peer" docker load
	fi
	REMOTE="$(fingerprint "$peer")"
	if [ "$REMOTE" = "$LOCAL" ]; then
		echo "$peer: OK   $REMOTE"
	else
		echo "$peer: MISMATCH  local=$LOCAL remote=$REMOTE" >&2
		exit 1
	fi
done
