import argparse
import yaml
import os
import sys
from src.core.train import train_model_from_config
# Note: We need to adapt src/core/train.py to accept a dictionary or config object instead of argparse options,
# or we convert the config dict to a namespace.

def main():
    parser = argparse.ArgumentParser(description="BCI Training Launcher")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # For now, we print the config to verify it loads
    print("Loaded configuration:")
    print(config)
    
    # TODO: Bridge this config to the legacy opt system in src/core/train.py
    # This requires modifying src/core/train.py to be callable.

if __name__ == "__main__":
    main()
