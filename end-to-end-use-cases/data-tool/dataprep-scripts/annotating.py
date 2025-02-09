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


def process_dataset(
    dataset,
    llm: LLM,
    system_prompt: str,
    output_file: str,
    start_index: int = 0,
    end_index: int = None,
    max_new_tokens: int = 128000,
) -> None:
    """Process the dataset using vLLM."""
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.95,
    )

    # Handle end_index
    if end_index is None:
        end_index = len(dataset)
    else:
        end_index = min(end_index, len(dataset))

    # Validate indices
    if start_index < 0:
        start_index = 0
    if start_index >= len(dataset):
        raise ValueError(
            f"Start index {start_index} is larger than dataset size {len(dataset)}"
        )
    if start_index >= end_index:
        raise ValueError(
            f"Start index {start_index} must be less than end index {end_index}"
        )

    # Select the specified range
    dataset_slice = dataset.select(range(start_index, end_index))

    # Process examples one at a time
    with open(output_file, "w") as f:
        for item in tqdm(
            dataset_slice, desc=f"Processing rows {start_index} to {end_index}"
        ):
            # Format the prompt as a single string
            prompt = format_prompt(system_prompt, item["conversations"])

            # Generate the response
            output = llm.generate(prompt, sampling_params)[0]

            print(output.outputs[0].text)
            # Save the result
            result = {
                "id": item["id"],
                "conversations": output.outputs[0].text,
            }
            f.write(json.dumps(result) + "\n")
