#!/usr/bin/env bash

# activate python environment
dataset="datasets/historical_wi"
run_name="test"

MASTER_ADDR=127.0.0.1
MASTER_PORT=12345

export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1

# run extraction script
./.venv/bin/torchrun \
    --nnodes=1 \
    --nproc-per-node=gpu \
    --standalone \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    scripts/extract_patches.py \
        --test-stride 224 \
        --num-windows 8192 \
        $dataset $run_name