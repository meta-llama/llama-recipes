# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.


import copy
from datasets import load_dataset
import itertools
import torch

# check system prompt token seq or user prompt token seq is in the current token list
def check_header(targets,seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] in targets:
            return True
    return False
def replace_target(target,seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] == target:
            seq[i],seq[i+1],seq[i+2] = -100,-100,-100
    return seq
def tokenize_dialogs(dialogs, images, processor):
    # If vocab size is above 128000, use the chat template to generate the tokens as it is from Llama 3 family models
    text_prompt = processor.apply_chat_template(dialogs)
    #print("text_prompt",text_prompt)
    batch = processor(images=images, text=text_prompt,padding = True, return_tensors="pt")
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i,n in enumerate(labels) if n == 128009]
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        prompt_header_seqs = [[128006, 9125, 128007],[128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx:idx+1]
            if check_header(prompt_header_seqs,current_seq):
                # found prompt header, indicating that this seq should be masked
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
            else:
                last_idx = idx+1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq,labels)
        # Mask the padding token and image token 128256 
        for i in range(len(labels)):
            if labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256: #  128256 is image token index
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    tokenizer_length = len(processor.tokenizer)
    return batch

def tokenize_dialog(dialog, images, processor):
    # If vocab size is above 128000, use the chat template to generate the tokens as it is from Llama 3 family models
    text_prompt = processor.apply_chat_template(dialog)
    #print("text_prompt",text_prompt)
    batch = processor(images=images, text=text_prompt,padding = True, return_tensors="pt")    
    labels = copy.copy(batch["input_ids"].tolist()[0])
    eot_indices = [i for i,n in enumerate(labels) if n == 128009]
    last_idx = 0
    # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
    # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
    prompt_header_seqs = [[128006, 9125, 128007],[128006, 882, 128007]]
    for n, idx in enumerate(eot_indices):
        current_seq = labels[last_idx:idx+1]
        if check_header(prompt_header_seqs,current_seq):
            # found prompt header, indicating that this seq should be masked
            labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
        else:
            last_idx = idx+1
        # Lastly mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
    assistant_header_seq = [128006, 78191, 128007]
    labels = replace_target(assistant_header_seq,labels)
    #print("labels",labels)
    # print("pixel_values .shape",batch["pixel_values"].shape)
    # print("batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape")

    batch["labels"] = torch.tensor(labels)
    # exit()
    # combined_tokens = {
    #     # "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
    #     # "labels": list(itertools.chain(*(t for t in labels_tokens))),
    #     "input_ids": dialog_tokens,
    #     "labels": labels,
    #     "attention_mask": [1]*len(dialog_tokens),
    #     "pixel_values": batch["pixel_values"],
    #     "aspect_ratio_ids": batch["aspect_ratio_ids"],
    #     "aspect_ratio_mask": batch["aspect_ratio_mask"],
    #     "cross_attention_mask": batch["cross_attention_mask"]
    # }
    # input_ids =  list(itertools.chain(*(t for t in dialog_tokens))),
    # labels = list(itertools.chain(*(t for t in labels_tokens))),
    # attention_mask = [1]*len(list(itertools.chain(*(t for t in dialog_tokens)))),
    # pixel_values =  batch["pixel_values"],
    # image_sizes = batch["image_sizes"]
#    print("combined_tokens",combined_tokens[image_sizes])
    
    return batch
def get_custom_dataset(dataset_config, processor, split, split_ratio=0.9):
    # load_dataset will return DatasetDict that contains all the data in the train set
    dataset_dict = load_dataset("remyxai/vqasynth_spacellava")
    dataset = dataset_dict[split]
    dataset = dataset.select(range(500))
    return dataset

class VQADataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right
    def __call__(self, samples):
        dialogs,images = [],[]
        for sample in samples:
            image,sample_text = sample["images"],sample["messages"]
            dialog = []
            for line in sample_text:
                content = []
                messages = line["content"]
                role = line["role"]
                for message in messages:
                    if message["type"] == "image":
                        content.append({"type": "image"})
                    elif message["type"] == "text":
                        content.append({"type": "text", "text": message["text"].strip()})
                dialog.append({"role": role,"content":content})
            dialogs.append(dialog)
            images.append(image)
        return tokenize_dialogs(dialogs,images, self.processor)
    def __callworking__(self, samples):
        for sample in samples:
            image,sample_text = sample["images"],sample["messages"]
            dialog = []
            for line in sample_text:
                content = []
                messages = line["content"]
                role = line["role"]
                for message in messages:
                    if message["type"] == "image":
                        content.append({"type": "image"})
                    elif message["type"] == "text":
                        content.append({"type": "text", "text": message["text"].strip()})
                dialog.append({"role": role,"content":content})
            return tokenize_dialog(dialog,image, self.processor)
def get_data_collator(processor):
    return VQADataCollator(processor)
