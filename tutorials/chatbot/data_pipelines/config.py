import yaml
import os


def load_config():
    # Read the YAML configuration file
    file_path = "./config.yaml"
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    config["api_key"] = os.getenv('OPENAI_API_KEY')
    return config
