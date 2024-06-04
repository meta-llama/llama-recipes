# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.


import copy
import datasets
from datasets import Dataset, load_dataset, DatasetDict
import itertools

B_INST, E_INST = "[INST]", "[/INST]"
def tokenize_dialog(dialog, tokenizer):
    # If vocab size is above 128000, use the chat template to generate the tokens as it is from Llama 3 family models
    if tokenizer.vocab_size >= 128000:
        dialog_tokens = tokenizer.apply_chat_template(dialog)
        dialog_tokens = dialog_tokens[:-4] # Remove generation prompt <|start_header_id|>assistant<|end_header_id|>\n\n
        eot_indices = [i for i,n in enumerate(dialog_tokens) if n == 128009]
        labels = copy.copy(dialog_tokens)
        last_idx = 0
        for n, idx in enumerate(eot_indices):
            if n % 2 == 1:
                last_idx = idx
            else:
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)

        dialog_tokens = [dialog_tokens]
        labels_tokens = [labels]
    else:
        # Otherwise, use the original tokenizer to generate the tokens as it is from Llama 2 family models
        prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog[:2]]
        answer = dialog[-1]
        answer_tokens = tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False)

        #Add labels, convert prompt token to -100 in order to ignore in loss function
        sample = {
            "input_ids": prompt_tokens + answer_tokens,
            "attention_mask" : [1] * (len(prompt_tokens) + len(answer_tokens)),
            "labels": [-100] * len(prompt_tokens) + answer_tokens,
            }
        return sample

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))
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
    chat = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
    {"role": "assistant", "content": answer}
    ]
    return tokenize_dialog(chat, tokenizer)


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
