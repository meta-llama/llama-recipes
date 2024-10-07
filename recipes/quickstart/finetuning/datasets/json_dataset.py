import copy
import torch
from datasets import Dataset
import os
import json
from PIL import Image

# Check if the system or user prompt header sequence is in the current token list
def check_header(targets, seq):
    for i in range(len(seq) - 3):
        if seq[i:i+3] in targets:
            return True
    return False

# Replace the target sequence with -100 (ignored index)
def replace_target(target, seq):
    for i in range(len(seq) - 3):
        if seq[i:i+3] == target:
            seq[i], seq[i+1], seq[i+2] = -100, -100, -100
    return seq

# Tokenize dialogs and prepare labels
def tokenize_dialogs(dialogs, images, processor):
    text_prompt = processor.apply_chat_template(dialogs)
    batch = processor(images=images, text=text_prompt, padding=True, return_tensors="pt")
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i, n in enumerate(labels) if n == 128009]
        last_idx = 0
        # System prompt header sequences
        prompt_header_seqs = [[128006, 9125, 128007], [128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx:idx+1]
            if check_header(prompt_header_seqs, current_seq):
                # Mask system and user prompt headers
                labels[last_idx:idx+1] = [-100] * (idx - last_idx + 1)
            else:
                last_idx = idx + 1
        # Assistant header sequence
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)
        # Mask padding token and image token (128256)
        for i in range(len(labels)):
            if labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256:
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch

# Function to load your custom dataset from a JSON file
def get_custom_dataset(dataset_config, processor, split, split_ratio=0.9):
    # Hardcoded dataset file path
    data_file = './Your_json_file_Name.json'  # Your JSON dataset file
    with open(data_file, 'r') as f:
        data = json.load(f)  # data is a list of dictionaries
    # Create a HuggingFace Dataset from the data
    dataset = Dataset.from_list(data)
    # Split the dataset into train and test sets
    dataset = dataset.train_test_split(test_size=1 - split_ratio, shuffle=True, seed=42)[split]
    return dataset

# Data collator class adjusted for your dataset's structure
class OCRVQADataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"  # During training, always use padding on the right
        # Hardcoded image path
        self.image_path = './data/images'  # Directory where your images are stored eg /home/myles/LLaVA-NeXT/data/images

    def __call__(self, samples):
        dialogs, images = [], []
        for sample in samples:
            image_filename = sample["image"]
            # Load the image from the specified image path
            image = Image.open(os.path.join(self.image_path, image_filename)).convert("RGB")
            conversations = sample["conversations"]
            dialog = []
            for idx, turn in enumerate(conversations):
                role = "user" if turn["from"] == "human" else "assistant"
                value = turn["value"].strip()
                if idx == 0 and "<image>" in value:
                    # Remove the <image> placeholder and add the image content
                    value = value.replace("<image>", "").strip()
                    dialog += [
                        {"role": role, "content": [{"type": "image"}, {"type": "text", "text": value}]}
                    ]
                else:
                    dialog += [
                        {"role": role, "content": [{"type": "text", "text": value}]}
                    ]
            dialogs.append(dialog)
            images.append([image])
        return tokenize_dialogs(dialogs, images, self.processor)

# Function to get the data collator
def get_data_collator(processor):
    return OCRVQADataCollator(processor)
