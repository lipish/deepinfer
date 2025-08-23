#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
WORKER_DIR="${ROOT_DIR}/python/worker"

cd "${WORKER_DIR}"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# generate python stubs
bash gen_proto.sh

export WORKER_ADDRESS=${WORKER_ADDRESS:-127.0.0.1:50051}
python server.py

