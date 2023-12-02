import datasets
from datasets import load_dataset
from .utils import Concatenator
import os

def get_preprocessed_uniphore(dataset_config, tokenizer, split):
    raw_train_dataset = load_dataset("json", data_files=os.path.join(os.environ["SM_CHANNEL_TRAIN"],"train.jsonl"), cache_dir="opt/ml/input", num_proc=8, keep_in_memory=True)['train']
    raw_validation_dataset = load_dataset("json", data_files=os.path.join(os.environ["SM_CHANNEL_TEST"],"val.jsonl"), cache_dir="opt/ml/input", num_proc=8, keep_in_memory=True)['train']
    # raw_train_dataset = raw_train_dataset.select(range(500))
    # raw_validation_dataset = raw_validation_dataset.select(range(50))
    dataset = datasets.DatasetDict({"train":raw_train_dataset,"validation":raw_validation_dataset})
    
    dataset_final = dataset[split].map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset[split].features)   
    ).map(Concatenator(chunk_size=4000), batched=True)
    
    return dataset_final
