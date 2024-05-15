# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import yaml
import os

def load_config(config_path: str = "./config.yaml"):
    # Read the YAML configuration file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    # Set the API key from the environment variable
    try:
        config["api_key"] = os.environ["OCTOAI_API_TOKEN"]
    except KeyError:
        print("API token did not found, please set the OCTOAI_API_TOKEN environment variable if using OctoAI, otherwise set api_key to default EMPTY")
        # local Vllm endpoint did not need API key, so set the API key to "EMPTY" if OCTOAI_API_TOKEN not found
        config["api_key"] = "EMPTY"
    return config
