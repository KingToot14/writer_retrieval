import subprocess

if __name__ == "__main__":
    # this is just a little helper script that runs a list of configuration files one after the other
    config_files = [
        "config/baseline_vits16.yaml",
        "config/grieggs.yaml",
        "config/pretrained_vits16.yaml",
    ]
    
    for config_file in config_files:
        subprocess.run(
            [
                ".venv/bin/python",
                "scripts/run_pipeline.py",
                config_file,
            ]
        )