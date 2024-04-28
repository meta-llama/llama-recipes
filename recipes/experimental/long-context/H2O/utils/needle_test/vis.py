'''
    Generate prompts for the LLM Needle Haystack.
    Source code from: 
        https://github.com/gkamradt/LLMTest_NeedleInAHaystack/tree/main
        https://github.com/THUDM/LongAlign/tree/main/Needle_test
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import os
import re
import glob
import argparse

if __name__ == '__main__':
    # Using glob to find all json files in the directory

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="")
    parser.add_argument("--exp-name", type=str, default="")
    args = parser.parse_args()

    json_files = glob.glob(f"{args.input_path}/*.json")

    if not os.path.exists('vis'):
        os.makedirs('vis')

    print(json_files)

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        print(file)
        # List to hold the data
        data = []

        with open(file, 'r') as f:
            json_data = json.load(f)
            
            for k in json_data:
                pattern = r"_len_(\d+)_"
                match = re.search(pattern, k)
                context_length = int(match.group(1)) if match else None

                pattern = r"depth_(\d+)"
                match = re.search(pattern, k)
                document_depth = eval(match.group(1))/100 if match else None

                score = json_data[k]['score']

                # Appending to the list
                data.append({
                    "Document Depth": document_depth,
                    "Context Length": context_length,
                    "Score": score
                })

        # Creating a DataFrame
        df = pd.DataFrame(data)

        pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
        pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
        
        # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

        # Create the heatmap with better aesthetics
        plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
        sns.heatmap(
            pivot_table,
            # annot=True,
            fmt="g",
            cmap=cmap,
            cbar_kws={'label': 'Score'},
            vmin=1,
            vmax=10,
        )

        # More aesthetics
        plt.title(f'Pressure Testing\nFact Retrieval Across Context Lengths ("Needle In A HayStack")\n{args.exp_name}')  # Adds a title
        plt.xlabel('Token Limit')  # X-axis label
        plt.ylabel('Depth Percent')  # Y-axis label
        plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
        plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
        plt.tight_layout()  # Fits everything neatly into the figure area
        # Show the plot
        plt.savefig(f"vis/{file.split('/')[-1].replace('.json', '')}.png")