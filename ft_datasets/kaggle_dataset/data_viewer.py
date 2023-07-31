import json
import kaggle_dataset
from transformers import LlamaTokenizer
from dataclasses import dataclass
import torch

@dataclass
class kd:
    dataset: str = "kaggle_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "/Users/adityagupta/kaggle_finetuning/data/v1_data/val/data.jsonl"

ACCESS_TOKEN = "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"
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf",   use_auth_token=ACCESS_TOKEN)
id = kaggle_dataset.InstructionDataset(kd, tokenizer)
torch.set_printoptions(profile="full")

vocab = tokenizer.get_vocab()
reverse_vocab = {v:k for k,v in vocab.items()}

print(len(id))
for idx in range(5):
    dp = id[idx]
    print(dp['input_ids'][:4001])
    print(dp['labels'][:4001])
    print(dp['attention_mask'][:4001])