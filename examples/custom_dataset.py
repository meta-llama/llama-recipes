# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets

from llama_recipes.datasets.utils import Concatenator

def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("OpenAssistant/oasst1", split=split)
    
    dataset = dataset.map(lambda sample: {
        "message_id": sample["message_id"],
        "parent_id": sample["parent_id"],
        "text": sample["text"],
        },
        batched=True,
        remove_columns=list(dataset.features),)
        
    # print(ids[0])
    
    p2c = {}
    
    ids2text = {}
    root_ids = []
    
    for data in dataset:
        if data["parent_id"]:
            p2c[data["parent_id"]] = p2c.get(data["parent_id"], []) + [data["message_id"]]
        else:
            root_ids.append(data["message_id"])
        ids2text[data["message_id"]]=data["text"]
           
    def follow(thread, current_id):
        thread = copy.copy(thread) + [ids2text[current_id]]
        if current_id in p2c:
            new_threads = []
            for next_id in p2c[current_id]:
                new_threads += follow(thread, next_id)
            return new_threads
        else:
            return [thread]
        
        
    def get_threads_from_root(root_id):
        all_threads = []
        thread = [ids2text[root_id]]
        for cid in p2c[root_id]:
            all_threads += follow(thread, cid)
        return all_threads
        
        
    # all_threads = []
    # for rid in root_ids:
        
            
    dataset = dataset.filter(lambda x: x["message_id"] in root_ids)
    dataset = dataset.map(lambda x: {"thread": get_threads_from_root(x["message_id"])}, remove_columns=list(dataset.features))
    dataset = dataset.map(lambda x: {"thread": [i for row in x["thread"] for i in row]}, batched=True)
            
    print(len(dataset))
    from pprint import pprint
    pprint(dataset[:10])
    
    return dataset
    # threads={}

    # prompt = (
    #     f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n{{summary}}{{eos_token}}"
    # )

    # def apply_prompt_template(sample):
    #     return {
    #         "text": prompt.format(
    #             dialog=sample["dialogue"],
    #             summary=sample["summary"],
    #             eos_token=tokenizer.eos_token,
    #         )
    #     }

    # dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
        
    # dataset = dataset.map(
    #     lambda sample: tokenizer(sample["text"]),
    #     batched=True,
    #     remove_columns=list(dataset.features),
    # ).map(Concatenator(), batched=True)
    # return dataset