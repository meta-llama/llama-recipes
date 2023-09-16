# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import pytest
from unittest.mock import patch


@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_custom_dataset(step_lr, optimizer, get_model, train, mocker):
    from llama_recipes.finetuning import main

    kwargs = {
        "dataset": "custom_dataset",
        "model_name": "decapoda-research/llama-7b-hf", # We use the tokenizer as a surrogate for llama2 tokenizer here
        "custom_dataset.file": "examples/custom_dataset.py",
        "custom_dataset.train_split": "validation",
        "batch_size_training": 2,
        "use_peft": False,
        }

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]
    tokenizer = args[3]

    assert len(train_dataloader) == 226
    assert len(eval_dataloader) == 2*226

    it = iter(train_dataloader)
    STRING = tokenizer.decode(next(it)["input_ids"][0], skip_special_tokens=True)
    EXPECTED_STRING = "[INST] Напиши функцию на языке swift, которая сортирует массив целых чисел, а затем выводит его на экран [/INST] Вот функция, "

    assert STRING.startswith(EXPECTED_STRING)

    next(it)
    next(it)
    next(it)
    STRING = tokenizer.decode(next(it)["input_ids"][0], skip_special_tokens=True)
    EXPECTED_SUBSTRING_1 = "Therefore you are correct.  [INST] How can L’Hopital’s Rule be"
    EXPECTED_SUBSTRING_2 = "a circular path around the turn.  [INST] How on earth is that related to L’Hopital’s Rule?"

    assert EXPECTED_SUBSTRING_1 in STRING
    assert EXPECTED_SUBSTRING_2 in STRING


@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.LlamaTokenizer.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_unknown_dataset_error(step_lr, optimizer, tokenizer, get_model, train, mocker):
    from llama_recipes.finetuning import main

    tokenizer.return_value = mocker.MagicMock(side_effect=lambda x: {"input_ids":[len(x)*[0,]], "attention_mask": [len(x)*[0,]]})

    kwargs = {
        "dataset": "custom_dataset",
        "custom_dataset.file": "examples/custom_dataset.py:get_unknown_dataset",
        "batch_size_training": 1,
        "use_peft": False,
        }
    with pytest.raises(AttributeError):
        main(**kwargs)

def test_system_prompt():
    from pathlib import Path
    from transformers import LlamaTokenizer

    from llama_recipes.utils.dataset_utils import load_module_from_py_file

    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

    module_path, func_name = "examples/custom_dataset.py", "tokenize_dialog"
    module_path = Path(module_path)
    module = load_module_from_py_file(module_path.as_posix())
    tokenize_dialog = getattr(module, func_name)

    system_prompt = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not "
        "make any sense, or is not factually coherent, explain why instead of answering something not correct. If you "
        "don't know the answer to a question, please don't share false information."
    )

    dialog = [
        {'content': "There's a llama in my garden 😱 What should I do?", 'role': 'user'},
    ]

    string = tokenizer.decode(tokenize_dialog(dialog, tokenizer, system_prompt)["input_ids"], skip_special_tokens=True)

    assert string == (
        "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
        "while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, "
        "or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a "
        "question does not make any sense, or is not factually coherent, explain why instead of answering something "
        "not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
        "There's a llama in my garden 😱 What should I do? [/INST]"
    )
