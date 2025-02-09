import argparse
import json
import os
from typing import Dict, List

import yaml
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams


def load_system_prompt(yaml_path: str) -> str:
    """Load system prompt from a YAML file."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config["system_prompt"]


def setup_llm(
    model_name: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 128000,
    gpu_ids: List[int] = None,
) -> LLM:
    """Initialize the vLLM LLM with specified parameters for multi-GPU support."""

    # If specific GPUs are requested, set CUDA_VISIBLE_DEVICES
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    return llm


def create_messages(system_prompt: str, conversation: str) -> List[Dict[str, str]]:
    """Create the messages list for the model input."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": conversation},
    ]


def format_prompt(system_prompt: str, conversation: str) -> str:
    """Format the system prompt and conversation into the specific chat template format."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>{conversation}"
    )


