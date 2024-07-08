# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.


import copy
from datasets import load_dataset
import itertools

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
def tokenize_dialog(dialog, tokenizer):
    # If vocab size is above 128000, use the chat template to generate the tokens as it is from Llama 3 family models
    if tokenizer.vocab_size >= 128000:
        dialog_tokens = tokenizer.apply_chat_template(dialog)
        eot_indices = [i for i,n in enumerate(dialog_tokens) if n == 128009]
        labels = copy.copy(dialog_tokens)
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
                last_idx = idx
        # Lastly mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq,labels)
        dialog_tokens = [dialog_tokens]
        labels_tokens = [labels]
    else:
        raise Exception("This raft_dataset only supports Llama 3 family models, please make sure the tokenizer is from Llama 3 family models.")

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))
def raft_tokenize(q_a_pair, tokenizer):
    end_tag = "</DOCUMENT>"
    # find the last end_tag in the instruction, the rest is the question
    try:
        index =q_a_pair["instruction"].rindex(end_tag)+len(end_tag)
    except ValueError:
        print(q_a_pair["instruction"])
        raise Exception("The instruction does not contain the end tag <\/DOCUMENT>")
    # all the lines after end_tag are the question
    question = q_a_pair["instruction"][index:].strip()
    # all the lines before end_tag are the context
    documents = q_a_pair["instruction"][:index].strip() 
    # output is the label
    answer = q_a_pair["output"]
    system_prompt = "You are a helpful chatbot who can provide an answer to every questions from the user given a relevant context."
    user_prompt = """
        Question: {question}\nContext: {context}\n
        Answer this question using the information given by multiple documents in the context above. Here are the things to pay attention to:
        - The context contains many documents, each document starts with <DOCUMENT> and ends </DOCUMENT>.
        - First provide step-by-step reasoning on how to answer the question.
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context.
        - End your response with final answer in the form <ANSWER>: $answer, the answer should less than 60 words.
        You MUST begin your final answer with the tag "<ANSWER>:".
    """.format(question=question, context=documents)

    chat = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
    {"role": "assistant", "content": answer}
    ]
    return tokenize_dialog(chat, tokenizer)


def get_custom_dataset(dataset_config, tokenizer, split, split_ratio=0.9):
    # load_dataset will return DatasetDict that contains all the data in the train set
    dataset_dict = load_dataset('json', data_files=dataset_config.data_path)
    dataset = dataset_dict['train']
    dataset = dataset.train_test_split(test_size=1-split_ratio, shuffle=True, seed=42)

    dataset = dataset[split].map(lambda sample: {
        "instruction": sample["instruction"],
        "output": sample["cot_answer"],
        },
        batched=True,
    )
    dataset = dataset.map(lambda x: raft_tokenize(x, tokenizer))
    return dataset
