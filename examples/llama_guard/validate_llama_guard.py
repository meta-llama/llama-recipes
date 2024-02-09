import json
from inference import llm_eval, AgentType
from typing import List, Tuple
#temp
import random

# Replace this with the actual path to your JSONL file
# TODO change to a parameter
jsonl_file_path = "valid_prompts_6cat_1122.jsonl"


def format_prompt(prompt, label, unsafe_category, idx):
    # TODO make this generic, depending on the file type?
    formatted_prompt = ([prompt], AgentType.USER, label, unsafe_category, idx)
    
    return formatted_prompt

# Initialize variables to accumulate the number of matches and mismatches
num_matches = 0
num_mismatches = 0
num_category_mismatches = 0
mismatched_indices = []
mismatched_categories_indices = []

# Iterate over each entry in the JSONL file
prompts: List[Tuple[List[str], AgentType, str, str, str]] = []
with open(jsonl_file_path, "r") as f:
    for idx, line in enumerate(f):
        entry = json.loads(line)
        
        # Call Llama Guard and get its output
        prompt = format_prompt(entry["prompt"], entry["label"], entry["unsafe_content"], entry["idx"])
        prompts.append(prompt)

#temp        
unsafe_list = [ 'safe',
'''unsafe
O1''', 
'''unsafe
# O2''',
'''unsafe
O3''',
'''unsafe
O4''',
'''unsafe
O5''',
'''unsafe
O6''']
# print(prompts)
results = [random.choice(unsafe_list) for _ in range(len(prompts))]

# results = llm_eval(prompts)

for i, result in enumerate(results):
    # Compare the output from Llama Guard against the unsafe_content column and the label content
    if prompts[i][2] == 'bad' and 'unsafe' not in result or prompts[i][2] == "good" and 'unsafe' in result:
        # TODO compare the categories as well

        num_mismatches += 1
        mismatched_indices.append(prompts[i][4])
        
        
    else:
        print(f"Result {result} - label {prompts[i][2]} categories {prompts[i][3]}")
        num_matches += 1

    if prompts[i][3] and prompts[i][3][0] not in result:
            num_category_mismatches += 1
            mismatched_categories_indices.append(prompts[i][4])

    

# Calculate the percentage of matches and print the results
# TODO change to a file output
percentage_matches = (num_matches / (num_matches + num_mismatches)) * 100
percentage_matches_categories = ((num_matches + num_mismatches - num_category_mismatches) / (num_matches + num_mismatches)) * 100

print(f"{num_matches} matches out of {num_matches + num_mismatches} ({percentage_matches:.2f}%)")
print(f"{num_matches + num_mismatches - num_category_mismatches} categorie matches out of {num_matches + num_mismatches} ({percentage_matches_categories:.2f}%)")
# print(f"Mismatched indices: {mismatched_indices}")
# print(f"Mismatched indices: {mismatched_categories_indices}")

not_in_b = list(set(mismatched_indices) - set(mismatched_categories_indices))
print(not_in_b)


# for idx in not_in_b:
#     print(list(filter(lambda prompt: prompt[4] == idx, prompts)))