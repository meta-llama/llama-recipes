# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import yaml

def load_config(config_path: str = "./config.yaml"):
    # Read the YAML configuration file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
