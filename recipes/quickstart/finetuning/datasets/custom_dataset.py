# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import itertools


B_INST, E_INST = "[INST]", "[/INST]"
EOT_ID = 128009 #<|eot_id|>

def mask_target(target,seq):
    for i in range(len(seq)-len(target)):
        if seq[i:i+len(target)] == target:
            seq[i:i+len(target)] = [-100] * len(target)
    return seq

def tokenize_dialog(dialog, tokenizer):
    if tokenizer.vocab_size >= 128000:
        dialog_tokens = tokenizer.apply_chat_template(dialog)
        eot_indices = [i for i,n in enumerate(dialog_tokens) if n == EOT_ID]
        labels = copy.copy(dialog_tokens)
        #determine token for system and user 
        system_or_user = (tokenizer.encode("system")[-1], tokenizer.encode("user")[-1])
        labels[0] = -100 # bos token
        last_idx = 1
        for n, idx in enumerate(eot_indices):
            role_token = labels[last_idx+1]
            if role_token in system_or_user:
                # Set labels to -100 for system and user tokens to ignore in loss function
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
            last_idx = idx + 1
        mask_target(tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False), labels)

        dialog_tokens = [dialog_tokens]
        labels_tokens = [labels]
    else:
        prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog[::2]]
        answer_tokens = [tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in dialog[1::2]]
        dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))

        #Add labels, convert prompt token to -100 in order to ignore in loss function
        labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("OpenAssistant/oasst1", split=split)

    dataset = dataset.map(lambda sample: {
        "message_id": sample["message_id"],
        "parent_id": sample["parent_id"],
        "text": sample["text"],
        },
        batched=True,
        remove_columns=list(dataset.features),)

    nodes = {}

    messages = {}
    root_ids = []

    for data in dataset:
        if data["parent_id"]:
            nodes[data["parent_id"]] = nodes.get(data["parent_id"], []) + [data["message_id"]]
        else:
            root_ids.append(data["message_id"])
        messages[data["message_id"]]=data["text"]

    def follow(thread, current_id):
        thread = copy.copy(thread) + [messages[current_id]]
        if current_id in nodes:
            new_threads = []
            for next_id in nodes[current_id]:
                new_threads += follow(thread, next_id)
            return new_threads
        else:
            return [thread]

    def get_threads_from_root(root_id):
        all_threads = []
        thread = [messages[root_id]]
        for cid in nodes[root_id]:
            all_threads += follow(thread, cid)
        return all_threads

    dataset = dataset.filter(lambda x: x["message_id"] in root_ids)
    dataset = dataset.map(lambda x: {"thread": get_threads_from_root(x["message_id"])}, remove_columns=list(dataset.features))
    dataset = dataset.map(lambda x: {"thread": [i for row in x["thread"] for i in row]}, batched=True)

    def to_dialog(thread):
        dialog = []
        for i, content in enumerate(thread):
            dialog.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": content,
            })
        return {"dialog": dialog}

    dataset = dataset.map(lambda x: to_dialog(x["thread"]), remove_columns=list(dataset.features))
    dataset = dataset.map(lambda x: tokenize_dialog(x["dialog"], tokenizer), remove_columns=list(dataset.features))

    return dataset
