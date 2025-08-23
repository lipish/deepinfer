#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${ROOT_DIR}"

export RUST_LOG=${RUST_LOG:-info}
export DEEPINFER_CONFIG=${DEEPINFER_CONFIG:-config/default.yaml}

cargo run -p api-server

