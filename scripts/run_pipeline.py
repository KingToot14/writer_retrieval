import os
from pathlib import Path

import yaml
from types import SimpleNamespace
from argparse import ArgumentParser

import subprocess

def dict_to_namespace(data: dict) -> SimpleNamespace:
    """
    A simple helper function that converts the `data` dictionary to a namespace that
    allows using dot notation for variable access
    """
    
    if isinstance(data, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in data.items()})
    
    elif isinstance(data, list):
        return [dict_to_namespace(v) for v in data]
    
    return data

if __name__ == "__main__":
    # create parser
    parser = ArgumentParser()
    parser.add_argument("config_file")
    
    # parse arguments
    args = parser.parse_args()
    
    config: SimpleNamespace
    
    with open(args.config_file, "r") as fs:
        config = dict_to_namespace(yaml.load(fs, Loader=yaml.FullLoader))
    
    # set environment variables
    repo = Path(__file__).resolve().parents[1]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    env["OMP_NUM_THREADS"] = str(config.runtime.omp_num_threads)
    env["NCCL_P2P_DISABLE"] = str(config.runtime.nccl_p2p_disable)
    env["CUDA_VISIBLE_DEVICES"] = str(config.runtime.cuda_visible_devices)
    
    # --- Patch Extraction --- #
    args = [
        "./.venv/bin/torchrun",
        f"--nnodes={config.runtime.nnodes}",
        f"--nproc-per-node={config.runtime.nproc_per_node}",
        "--standalone",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=localhost:0",
        "scripts/extract.py",
    ]
    
    script_args = [
        "--train-stride", str(config.extract.train_stride),
        "--test-stride", str(config.extract.test_stride),
        "--num-windows", str(config.extract.num_windows),
        "--weights", str(config.weights),
    ]

    # check DINOv1
    if config.model.kind == "dino_v1":
        args.append("--dino-v1")

    # add positional args
    script_args += [
        config.dataset, config.run_name,
    ]
    
    args += script_args

    # --- Model Training --- #
    subprocess.run(
        args,
        cwd=repo,
        env=env,
        check=True
    )
    
    args = [
        "./.venv/bin/python",
        "scripts/train_models.py",
        f"output/patches/{config.run_name}", config.run_name,
    ]
    
    # --- Metrics --- #
    subprocess.run(
        args,
        cwd=repo,
        env=env,
        check=True
    )
    
    args = [
        "./.venv/bin/python",
        "scripts/get_metrics.py",
        f"output/patches/{config.run_name}", config.run_name,
    ]
    
    subprocess.run(
        args,
        cwd=repo,
        env=env,
        check=True
    )