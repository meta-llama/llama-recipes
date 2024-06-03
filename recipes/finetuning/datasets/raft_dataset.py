# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.


import copy
import datasets
from datasets import Dataset, load_dataset, DatasetDict
import itertools


B_INST, E_INST = "[INST]", "[/INST]"

def raft_tokenize(q_a_pair, tokenizer):
    # last line is the question
    question = q_a_pair["instruction"].split('\n')[-1]
    # all the lines before the last line are the context
    documents = q_a_pair["instruction"].split('\n')[:-1]
    # output is the label
    answer = q_a_pair["output"]
    system_prompt = "You are a helpful question answerer who can provide an answer given a question and relevant context."
    user_prompt = prompt = """
        Question: {question}\nContext: {context}\n
        Answer this question using the information given in the context above. Here is things to pay attention to:
        - First provide step-by-step reasoning on how to answer the question.
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context.
        - End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct.
        You MUST begin your final answer with the tag "<ANSWER>:".
    """.format(question=question, context=str(documents))
    final_prompt = system_prompt + '\n' + user_prompt
    prompt_tokens = tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(final_prompt).strip()} {E_INST}", add_special_tokens=False)
    answer_tokens = tokenizer.encode(f"{answer.strip()} {tokenizer.eos_token}", add_special_tokens=False)
    #Add labels, convert prompt token to -100 in order to ignore in loss function
    sample = {
            "input_ids": prompt_tokens + answer_tokens,
            "attention_mask" : [1] * (len(prompt_tokens) + len(answer_tokens)),
            "labels": [-100] * len(prompt_tokens) + answer_tokens,
            }

    return sample


def get_custom_dataset(dataset_config, tokenizer, split, split_ratio=0.8):
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
