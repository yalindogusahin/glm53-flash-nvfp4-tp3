#!/usr/bin/env bash
#
# Fetch GLM-5.3-Flash-NVFP4 (~182 GiB, 120 shards) plus the chat template.
#
#   fetch-weights.sh <target-dir> [repo-id]
#
# The chat template is a separate file in the HF repo, so weight-oriented downloaders skip
# it - and transformers >= 4.44 refuses to serve without one. Fetch it explicitly.
set -euo pipefail

DEST="${1:?usage: fetch-weights.sh <target-dir> [repo-id]}"
REPO="${2:-LibertAIDAI/GLM-5.3-Flash-NVFP4}"

mkdir -p "$DEST"
python3 - "$REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, local_dir=dest, max_workers=8,
                  allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model"])
print("downloaded", repo, "->", dest)
PY

if [ ! -f "$DEST/chat_template.jinja" ]; then
	curl -fsSL -o "$DEST/chat_template.jinja" \
		"https://huggingface.co/$REPO/resolve/main/chat_template.jinja"
	echo "fetched chat_template.jinja separately"
fi

python3 - "$DEST" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
shards = len(list(d.glob("*.safetensors")))
size = sum(f.stat().st_size for f in d.glob("*.safetensors")) / 2**30
cfg = json.loads((d / "config.json").read_text())["text_config"]
print(f"{shards} shards, {size:.1f} GiB, heads={cfg['num_attention_heads']}, "
      f"moe_i={cfg['moe_intermediate_size']}, template={'yes' if (d/'chat_template.jinja').exists() else 'MISSING'}")
print("next: scripts/pad-config.py", d / "config.json", "--tp 3")
PY
