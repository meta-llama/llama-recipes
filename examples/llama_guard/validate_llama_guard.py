import json
from examples.llama_guard.inference import llm_eval, standard_llm_eval, AgentType
from typing import List, Tuple
import fire
from enum import Enum

from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.metrics import average_precision_score

class Type(Enum):
    HF = "HF"
    PYTORCH = "PYTORCH"

def validate_parameters(agent_type, type, ckpt_dir):
    validate_agent_type(agent_type)
    if type is Type.PYTORCH.value and ckpt_dir is None:
        raise ValueError(f"ckpt_dir cannot be null when type is {Type.PYTORCH.value}. Provide a path to the llama guard checkpoints")
        

def validate_agent_type(value):
    try:
        return AgentType(value)
    except ValueError:
        raise ValueError(f"Invalid AgentType. Choose from: {[agent_type.value for agent_type in AgentType]}")

def format_prompt(entry, agent_type: AgentType):
    prompts = []
    if agent_type == AgentType.USER:
        prompts = [entry["prompt"]]
    else:
        prompts = [entry["prompt"], entry["generation"]]

    return {
        "prompt": prompts,
        "agent_type": agent_type,
        "label": entry["label"],
        "unsafe_content": entry["unsafe_content"],
        "idx": entry["idx"]
    }

def is_label_mismatch(label, result):
    return (label == 'bad' and 'unsafe' not in result) or (label == 'good' and 'unsafe' in result)

def is_category_mismatch(categories, result):
    return categories and categories[0] not in result

def is_label_bad_and_unsafe(label, result):
    return label == "bad" and "unsafe" in result

def parse_results(prompts):

    # Initialize variables to accumulate the number of matches and mismatches
    num_matches = 0
    num_mismatches = 0
    num_total_unsafe = 0
    num_category_mismatches = 0
    mismatched_indices = []
    mismatched_categories_indices = []

    for i, prompt in enumerate(prompts):
        label = prompt["label"]
        categories = prompt["unsafe_content"]
        index = prompt["idx"]
        # Compare the output from Llama Guard against the label content
        if is_label_mismatch(label, prompt["result"]):
            num_mismatches += 1
            mismatched_indices.append(index)
        else:
            num_matches += 1

        # Count only unsafe entries to check the percentage of categories missclasified
        if label == "bad":
            num_total_unsafe += 1

        # Compare the output from Llama Guard against the unsafe_content column
        if is_label_bad_and_unsafe(label, prompt["result"]) and is_category_mismatch(categories, prompt["result"]):
            num_category_mismatches += 1
            mismatched_categories_indices.append(prompt["idx"])

    return {
        "num_matches": num_matches,
        "num_mismatches": num_mismatches,
        "num_total_unsafe": num_total_unsafe,
        "num_category_mismatches": num_category_mismatches,
        "mismatched_indices": mismatched_indices,
        "mismatched_categories_indices": mismatched_categories_indices
    }

# def parse_logprobs(result_logprobs):
def parse_logprobs(prompts):
    if isinstance(prompts[0]["logprobs"][0], float):
        positive_class_probs = [np.exp(prompt["logprobs"][0]) for prompt in prompts]
    else:
        positive_class_probs = [np.exp(prompt["logprobs"][0][1]) for prompt in prompts]
    binary_labels = [1 if prompt["label"] == "bad" else 0 for prompt in prompts]
    return average_precision_score(binary_labels, positive_class_probs)


def main(jsonl_file_path, agent_type, type: Type, ckpt_dir = None, load_in_8bit: bool = True, logprobs: bool = True):

    input_file_path = Path(jsonl_file_path)

    # Extracting filename without extension
    filename = input_file_path.stem
    directory = input_file_path.parent

    validate_parameters(agent_type, type, ckpt_dir)
    agent_type = AgentType(agent_type)
    type = Type(type)

    # Preparing prompts
    prompts: List[Tuple[List[str], AgentType, str, str, str]] = []
    with open(jsonl_file_path, "r") as f:
        # temp
        index = 0 
        for i, line in enumerate(f):
            if index == 10:
                break
            index += 1

            entry = json.loads(line)
            
            # Call Llama Guard and get its output
            prompt = format_prompt(entry, agent_type)
            prompts.append(prompt)
            
            

    # Executing evaluation
    if type is Type.HF:
        llm_eval(prompts, load_in_8bit, logprobs)
    else:
        standard_llm_eval(prompts, ckpt_dir, logprobs)
    
    # if logprobs:
    #     for result in results[1]:
    #         for tuple_result in result:
    #             # | token | log probability | probability
    #             print(f"| {tuple_result[0]:5d} | {tuple_result[1]:.9f} | {np.exp(tuple_result[1]):.2%}\n")
                # pass

    parsed_results = parse_results(prompts)
    if logprobs:
        average_precision = parse_logprobs(prompts)
        parsed_results["average_precision"] = average_precision

    # Calculate the percentage of matches, print the results and store for file output
    num_matches = parsed_results["num_matches"]
    num_mismatches = parsed_results["num_mismatches"]
    num_total_unsafe = parsed_results["num_total_unsafe"]
    num_category_mismatches = parsed_results["num_category_mismatches"]
    
    percentage_matches = (num_matches / (num_matches + num_mismatches)) * 100
    percentage_matches_categories = ((num_total_unsafe - num_category_mismatches) / (num_total_unsafe)) * 100
    parsed_results["percentage_matches"] = percentage_matches
    parsed_results["percentage_matches_categories"] = percentage_matches_categories


    print(f"{num_matches} matches out of {num_matches + num_mismatches} ({percentage_matches:.2f}%)")
    print(f"{num_total_unsafe - num_category_mismatches} categorie matches out of {num_total_unsafe} ({percentage_matches_categories:.2f}%)")
    print(f"average precision {average_precision:.2%}")

    # Output filenames and paths
    current_date = datetime.now().strftime('%Y-%m-%d-%H:%M')
    output_filename = f"{filename}_{agent_type.value}_{type.value}_{'load_in_8bit' if load_in_8bit else 'noquant'}_output_{current_date}.jsonl"
    output_stats_filename = f"{filename}_{agent_type.value}_{type.value}_{'load_in_8bit' if load_in_8bit else 'noquant'}_stats_{current_date}.json"
    output_file_path = directory / output_filename
    output_stats_file_path = directory / output_stats_filename

    # Write the list of dictionaries to the file
    with open(output_file_path, 'w') as file:
        for prompt in prompts:
            # Serialize each dictionary to a JSON formatted string
            prompt.pop("logprobs", None)
            json_str = json.dumps(prompt, default=lambda o: o.value if isinstance(o, Enum) else o)
            # Write the JSON string to the file followed by a newline
            file.write(json_str + '\n')
        
    # Write the stats from the run
    with open(output_stats_file_path, 'w') as f:
        json.dump(parsed_results, f, indent=4)


if __name__ == '__main__':
    fire.Fire(main)