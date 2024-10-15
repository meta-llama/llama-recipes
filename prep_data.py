import fire
from transformers import (
    LlamaTokenizer,
)

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.data.concatenator import ConcatDataset

from llama_recipes.utils.config_utils import (
    update_config,
    generate_dataset_config,
)

from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)

    dataset_config = generate_dataset_config(train_config, kwargs)
    print("dataset_config")

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(kwargs["model_path"])
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )
    print(f"--> Training Set Length = {len(dataset_train)}")
    # for default, it gets return from:
    # get_preprocessed_samsum(
    #   dataset_config,
    #   tokenizer,
    #   dataset_config.train_split (which is just 'train')
    # )

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    print(f"--> Validation Set Length = {len(dataset_val)}")

    if train_config.batching_strategy == "packing":
        print("dataset batching_strategy == packing")
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)
        dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)
        print("dataset = ConcatDataset")

if __name__ == "__main__":
    fire.Fire(main)