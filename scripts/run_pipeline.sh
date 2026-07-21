#!/usr/bin/env bash

main() {
    dataset=$1
    run_name=$2
    weights=$3
    use_dino_v1=$4

    # check parameters
    if [ -z $dataset ]; then
        echo "'dataset' must not be empty"
        return
    fi
    if [ -z $run_name ]; then
        echo "'run_name' must not be empty"
        return
    fi
    if [ -z $weights ]; then
        echo "Using default weights: 'weights/dinov3_vits16'"
        weights='weights/dinov3_vits16'
    fi

    # set torchrun parameters
    MASTER_ADDR=127.0.0.1
    MASTER_PORT=12345

    export NCCL_P2P_DISABLE=1
    export CUDA_VISIBLE_DEVICES=0,1
    export PYTHONPATH=$PWD
    export OMP_NUM_THREADS=1

    # run extraction script
    echo "[Writer Retrieval] Extracting all patch tokens"

    if [ -z $use_dino_v1 ]; then
        echo "Using model: 'DINOv3'"
        ./.venv/bin/torchrun \
            --nnodes=1 \
            --nproc-per-node=gpu \
            --standalone \
            --rdzv-backend=c10d \
            --rdzv-endpoint=localhost:0 \
            scripts/extract.py \
                --test-stride 56 \
                --num-windows 4096 \
                --weights $weights \
                $dataset $run_name
    else
        echo "Using model: 'DINOv1'"
        ./.venv/bin/torchrun \
            --nnodes=1 \
            --nproc-per-node=gpu \
            --standalone \
            --rdzv-backend=c10d \
            --rdzv-endpoint=localhost:0 \
            scripts/extract.py \
                --test-stride 56 \
                --num-windows 4096 \
                --weights $weights \
                --dino-v1 \
                $dataset $run_name
    fi

    # run vlad training
    echo "[Writer Retrieval] Training VLAD codebook and PCA Matrix"

    ./.venv/bin/python \
        scripts/train_models.py \
            output/patches/$run_name $run_name

    # run metrics
    echo "[Writer Retrieval] Collecting metrics on VLAD descriptors"

    ./.venv/bin/python \
        scripts/get_metrics.py \
            output/patches/$run_name $run_name
}

main $1 $2 $3 $4