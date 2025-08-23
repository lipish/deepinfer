#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="${SCRIPT_DIR}/../../proto"

python -m grpc_tools.protoc \
  -I"${PROTO_DIR}" \
  --python_out="${SCRIPT_DIR}" \
  --grpc_python_out="${SCRIPT_DIR}" \
  "${PROTO_DIR}/engine/v1/engine.proto"

echo "Generated Python stubs under ${SCRIPT_DIR}/engine/v1"
