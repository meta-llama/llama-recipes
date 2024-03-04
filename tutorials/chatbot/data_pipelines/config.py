# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import yaml
import os

def load_config():
    # Read the YAML configuration file
    file_path = "./config.yaml"
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    # Set the API key from the environment variable
    config["api_key"] = os.environ["OCTOAI_API_TOKEN"]
    return config
