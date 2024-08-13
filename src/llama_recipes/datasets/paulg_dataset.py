import pandas as pd
from datasets import Dataset
import re

# from transformers import AutoTokenizer

IGNORE_LOSS_ID = -100

# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt_completions_path = "data.json"

def get_preprocessed_paulg(dataset_config, tokenizer, split) -> Dataset:
    df = pd.read_json(prompt_completions_path)
    # df["answer"] = df["answer"].str.replace(" \n", "\n")
    # df["answer"] = df["answer"].str.replace("\n ", "\n")
    # df["answer"] = df["answer"].apply(lambda x: re.sub(r'\n+', '\n', x))
    # df["answer"] = df["answer"].str.strip()

    if split == "train":
        df = df[:-10]
    else:
        df = df[-10:]
    print(len(df))

    dataset = Dataset.from_pandas(df)

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["question"], add_special_tokens=False)
        response = tokenizer.encode(sample["answer"] +  tokenizer.eos_token, add_special_tokens=False)
        return {
            "input_ids": prompt + response,
            "attention_mask" : [1] * (len(prompt) + len(response)),
            "labels": [IGNORE_LOSS_ID] * len(prompt) + response,
        }

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    print(len(dataset))
    return dataset
