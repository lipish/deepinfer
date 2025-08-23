#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${ROOT_DIR}"

# Download GPT-2 tokenizer.json via Hugging Face (requires git-lfs and network)
mkdir -p config
URL="https://huggingface.co/gpt2/resolve/main/tokenizer.json?download=true"
OUT="config/tokenizer.json"

if command -v curl >/dev/null 2>&1; then
  curl -L "$URL" -o "$OUT"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$OUT" "$URL"
else
  echo "Please install curl or wget to fetch tokenizer.json" >&2
  exit 1
fi

echo "Saved to $OUT"

