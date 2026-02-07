#!/bin/bash

# Start etcd in single-node mode for development/testing

ETCD_DIR="/home/ai/lipeng/deepinfer"
DATA_DIR="${ETCD_DIR}/data/etcd"

mkdir -p "${DATA_DIR}"

echo "Starting etcd on localhost:2379..."

"${ETCD_DIR}/etcd" \
  --name deepinfer-etcd \
  --data-dir "${DATA_DIR}" \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://localhost:2379 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --initial-advertise-peer-urls http://localhost:2380 \
  --initial-cluster deepinfer-etcd=http://localhost:2380 \
  --initial-cluster-state new \
  --log-level info
