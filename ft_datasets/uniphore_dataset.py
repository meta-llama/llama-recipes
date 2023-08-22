import datasets
from datasets import load_dataset
from .utils import Concatenator
import os

def get_preprocessed_uniphore(dataset_config, tokenizer, split):
    
    raw_train_dataset = load_dataset("json", data_files=os.path.join(dataset_config.data_path,"train.jsonl"))['train']
    #os.path.join(os.environ["SM_CHANNEL_TRAIN"],"train.jsonl")['train'] #dataset_config.data_path))
    raw_validation_dataset = load_dataset("json", data_files=os.path.join(dataset_config.data_path,"val.jsonl"))['train']
    
    dataset = datasets.DatasetDict({"train":raw_train_dataset,"validation":raw_validation_dataset})
    
    # Define a function to process each sample
    def process_sample(sample):

        # Merge the modified input with the target
        merged_text = sample["input"] + "\n\nSummary:\n" + sample["target"] + tokenizer.eos_token

        # Return the merged text as a new sample
        return {"text": merged_text}

    # Apply the process_sample function using map()
    processed_dataset = dataset.map(process_sample,remove_columns=['input', 'target'])
    
    dataset_final = processed_dataset[split].map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(processed_dataset[split].features),
    ).map(Concatenator(chunk_size=4000), batched=True)
    
    return dataset_final