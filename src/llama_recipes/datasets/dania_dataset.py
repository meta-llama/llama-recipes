import copy
import json

import torch
from torch.utils.data import Dataset

import llama_recipes.data.llama_guard.finetuning_data_formatter

INST_TOKEN_SIZE=7
NUM_SAMPLES_TEST=200

class DaniaDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):   
        self.annotated_data = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.annotated_data  = dict(list(self.annotated_data.items())[NUM_SAMPLES_TEST:])
                  
        else:
             self.annotated_data  = dict(list(self.annotated_data.items())[:NUM_SAMPLES_TEST])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.annotated_data)
 
    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        full_prompt = list(self.annotated_data.items())[index][1]
                
        # Prompt length
        index_of_instr_enc = full_prompt.find("[/INST]")
        
        prompt = full_prompt[:index_of_instr_enc+INST_TOKEN_SIZE] # Size of /INST
       
        
        output = full_prompt[index_of_instr_enc+INST_TOKEN_SIZE:-1]
        example = prompt + output
        
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
       
       
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        
      

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }

