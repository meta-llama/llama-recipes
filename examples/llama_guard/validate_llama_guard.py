import json
from examples.llama_guard.inference import llm_eval, AgentType
from typing import List, Tuple
import fire

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

    return (prompts, agent_type, entry["label"], entry["unsafe_content"], entry["idx"])

def is_label_mismatch(label, result):
    return (label == 'bad' and 'unsafe' not in result) or (label == 'good' and 'unsafe' in result)

def is_category_mismatch(categories, result):
    return categories and categories[0] not in result

def is_label_bad_and_unsafe(label, result):
    return label == "bad" and "unsafe" in result

def parse_results(results, prompts):

    num_matches = 0
    num_mismatches = 0
    num_category_mismatches = 0
    mismatched_indices = []
    mismatched_categories_indices = []

    for i, result in enumerate(results):
        label = prompts[i][2]
        categories = prompts[i][3]
        index = prompts[i][4]
        # Compare the output from Llama Guard against the unsafe_content column and the label content
        if is_label_mismatch(label, result):
            num_mismatches += 1
            mismatched_indices.append(index)
        else:
            num_matches += 1

        if is_label_bad_and_unsafe(label, result) and is_category_mismatch(categories, result):
            num_category_mismatches += 1
            mismatched_categories_indices.append(prompts[i][4])

    return {
        "num_matches": num_matches,
        "num_mismatches": num_mismatches,
        "num_category_mismatches": num_category_mismatches,
        "mismatched_indices": mismatched_indices,
        "mismatched_categories_indices": mismatched_categories_indices
    }

def main(jsonl_file_path, agent_type, load_in_8bit = True):

    agent_type = validate_agent_type(agent_type)
    # Initialize variables to accumulate the number of matches and mismatches
    prompts: List[Tuple[List[str], AgentType, str, str, str]] = []
    with open(jsonl_file_path, "r") as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            
            # Call Llama Guard and get its output
            prompt = format_prompt(entry, agent_type)
            prompts.append(prompt)

    results = llm_eval(prompts, load_in_8bit)

    parsed_results = parse_results(results, prompts)
   
    num_matches = parsed_results["num_matches"]
    num_mismatches = parsed_results["num_mismatches"]
    num_category_mismatches = parsed_results["num_category_mismatches"]
    mismatched_indices = parsed_results["mismatched_indices"]
    mismatched_categories_indices = parsed_results["mismatched_categories_indices"]

    # Calculate the percentage of matches and print the results
    # TODO change to a file output
    percentage_matches = (num_matches / (num_matches + num_mismatches)) * 100
    percentage_matches_categories = ((num_matches + num_mismatches - num_category_mismatches) / (num_matches + num_mismatches)) * 100

    print(f"{num_matches} matches out of {num_matches + num_mismatches} ({percentage_matches:.2f}%)")
    print(f"{num_matches + num_mismatches - num_category_mismatches} categorie matches out of {num_matches + num_mismatches} ({percentage_matches_categories:.2f}%)")
    # print(f"Mismatched indices: {mismatched_indices}")
    # print(f"Mismatched indices: {mismatched_categories_indices}")


if __name__ == '__main__':
    fire.Fire(main)