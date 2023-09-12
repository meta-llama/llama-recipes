import datasets
from datasets import load_dataset
from .utils import Concatenator
import os

def get_preprocessed_uniphore(dataset_config, tokenizer, split):
    
    raw_train_dataset = load_dataset("json", data_files=os.path.join(os.environ["SM_CHANNEL_TRAIN"],"train.jsonl"))['train']
    raw_validation_dataset = load_dataset("json", data_files=os.path.join(os.environ["SM_CHANNEL_TEST"],"val.jsonl"))['train']
    dataset = datasets.DatasetDict({"train":raw_train_dataset,"validation":raw_validation_dataset})
    
    dataset_final = dataset[split].map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset[split].features),
    ).map(Concatenator(chunk_size=4000), batched=True)
    
    return dataset_final
