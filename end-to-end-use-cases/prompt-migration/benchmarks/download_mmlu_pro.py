from datasets import load_dataset
import pandas as pd
import os

def download_mmlu_pro():
    # Create output directory if it doesn't exist
    output_dir = "mmlu_pro_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    
    # Convert each split to CSV
    for split in dataset.keys():
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset[split])
        
        # Save to CSV
        output_path = os.path.join(output_dir, f"mmlu_pro_{split}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {split} split to {output_path}")
        print(f"Number of examples in {split}: {len(df)}")
        
if __name__ == "__main__":
    print("Downloading MMLU-Pro dataset...")
    download_mmlu_pro()
    print("Download complete!")
