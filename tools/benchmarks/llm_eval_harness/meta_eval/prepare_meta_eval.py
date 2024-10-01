# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.

import argparse
import errno
import shutil
import glob
import os
from pathlib import Path
import nltk
import yaml
from datasets import Dataset, load_dataset


# get the ifeval  from the evals dataset and join it with the original ifeval datasets
def get_ifeval_data(model_name, output_dir):
    print(f"preparing the ifeval data using {model_name}'s evals dataset")
    if model_name not in [
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "Llama-3.1-405B-Instruct",
    ]:
        raise ValueError(
            "Only Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct, Llama-3.1-405B-Instruct models are supported for IFEval"
        )
    original_dataset_name = "wis-k/instruction-following-eval"
    meta_dataset_name = f"meta-llama/{model_name}-evals"
    meta_data = load_dataset(
        meta_dataset_name,
        name=f"{model_name}-evals__ifeval__strict__details",
        split="latest",
    )
    ifeval_data = load_dataset(original_dataset_name, split="train")
    meta_data = meta_data.map(get_question)
    meta_df = meta_data.to_pandas()
    ifeval_df = ifeval_data.to_pandas()
    ifeval_df = ifeval_df.rename(columns={"prompt": "input_question"})
    # join the two datasets on the input_question column
    joined = meta_df.join(ifeval_df.set_index("input_question"), on="input_question")
    joined = joined.rename(columns={"input_final_prompts": "prompt"})
    joined = joined.rename(columns={"is_correct": "previous_is_correct"})
    joined = Dataset.from_pandas(joined)
    joined = joined.select_columns(
        [
            "input_question",
            "prompt",
            "previous_is_correct",
            "instruction_id_list",
            "kwargs",
            "output_prediction_text",
            "key",
        ]
    )
    joined.rename_column("output_prediction_text", "previous_output_prediction_text")
    joined.to_parquet(output_dir + "/joined_ifeval.parquet")


# get the math_hard data from the evals dataset and join it with the original math_hard dataset
def get_math_data(model_name, output_dir):
    print(f"preparing the math data using {model_name}'s evals dataset")
    if model_name not in [
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "Llama-3.1-405B-Instruct",
    ]:
        raise ValueError(
            "Only Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct, Llama-3.1-405B-Instruct models are supported for MATH_hard"
        )
    original_dataset_name = "lighteval/MATH-Hard"
    meta_dataset_name = f"meta-llama/{model_name}-evals"
    meta_data = load_dataset(
        meta_dataset_name,
        name=f"{model_name}-evals__math_hard__details",
        split="latest",
    )
    math_data = load_dataset(original_dataset_name, split="test")
    meta_df = meta_data.to_pandas()
    math_df = math_data.to_pandas()
    math_df = math_df.rename(columns={"problem": "input_question"})
    # join the two datasets on the input_question column
    joined = meta_df.join(math_df.set_index("input_question"), on="input_question")
    joined = Dataset.from_pandas(joined)
    joined = joined.select_columns(
        [
            "input_question",
            "input_correct_responses",
            "input_final_prompts",
            "is_correct",
            "solution",
            "output_prediction_text",
        ]
    )
    joined = joined.rename_column("is_correct", "previous_is_correct")
    joined = joined.rename_column(
        "output_prediction_text", "previous_output_prediction_text"
    )

    joined.to_parquet(output_dir + "/joined_math.parquet")


# get the question from the ifeval dataset
def get_question(example):
    try:
        example["input_question"] = (
            eval(
                example["input_question"]
                .replace("null", "None")
                .replace("true", "True")
                .replace("false", "False")
            )["dialog"][0]["body"]
            .replace("Is it True that the first song", "Is it true that the first song")
            .replace("Is the following True", "Is the following true")
        )
        example["input_final_prompts"] = example["input_final_prompts"][0]
        return example
    except:
        print(example["input_question"])
        return


# change the yaml file to use the correct model name
def change_yaml(args, base_name):
    for yaml_file in glob.glob(args.template_dir + "**/*/*.yaml", recursive=True):
        with open(yaml_file, "r") as sources:
            lines = sources.readlines()
        output_path = yaml_file.replace(args.template_dir, args.work_dir)
        print(f"changing {yaml_file} to output_path: {output_path}")
        path = Path(output_path)
        yaml_dir = path.parent
        with open(output_path, "w") as output:
            for line in lines:
                output.write(
                    line.replace("Llama-3.1-8B", base_name).replace(
                        "WORK_DIR", str(yaml_dir)
                    )
                )


# copy the files and change the yaml file to use the correct model name
def copy_and_prepare(args):
    # nltk punkt_tab package is needed
    nltk.download('punkt_tab')
    if not os.path.exists(args.work_dir):
        # Copy the all files, including yaml files and python files, from template folder to the work folder

        copy_dir(args.template_dir, args.work_dir)
    else:
        print("work_dir already exists, no need to copy files")
    # Use the template yaml to get the correct model name in work_dir yaml
    base_name = (
        args.evals_dataset.split("/")[-1].replace("-evals", "").replace("-Instruct", "")
    )
    change_yaml(args, base_name)


def parse_eval_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config_path",
        type=str,
        default="./eval_config.yaml",
        help="the config yaml file that contains all the eval parameters",
    )
    return parser.parse_args()


def prepare_datasets(args):
    # Prepare the dataset for the IFeval and MATH_Hard tasks as we need to join the original dataset with the evals dataset by the actual questions.
    # model_name are derived from the evals_dataset name
    task_list = args.tasks.split(",")
    model_name = args.evals_dataset.split("/")[-1].replace("-evals", "")
    if "meta_instruct" in task_list:
        get_ifeval_data(model_name, args.work_dir)

        get_math_data(model_name, args.work_dir)
    else:
        if "meta_ifeval" in task_list:
            get_ifeval_data(model_name, args.work_dir)
        if "meta_math_hard" in task_list:
            get_math_data(model_name, args.work_dir)


# copy the files from src to dst
def copy_dir(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc:  # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else:
            raise


# load the config yaml file
def load_config(config_path: str = "./config.yaml"):
    # Read the YAML configuration file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    args = parse_eval_args()
    config = load_config(args.config_path)
    # Create VLLM model args
    for k, v in config.items():
        args.__setattr__(k, v)
    if not os.path.exists(args.template_dir):
        raise ValueError("The template_dir does not exist, please check the path")
    if args.evals_dataset not in [
        "meta-llama/Llama-3.1-8B-Instruct-evals",
        "meta-llama/Llama-3.1-70B-Instruct-evals",
        "meta-llama/Llama-3.1-405B-Instruct-evals",
        "meta-llama/Llama-3.1-8B-evals",
        "meta-llama/Llama-3.1-70B-evals",
        "meta-llama/Llama-3.1-405B-evals",
    ]:
        raise ValueError(
            "The evals dataset is not valid, please double check the name, must use the name in the Llama 3.1 Evals collection"
        )
    args.model_args = f"pretrained={args.model_name},tensor_parallel_size={args.tensor_parallel_size},dtype=auto,gpu_memory_utilization={args.gpu_memory_utilization},data_parallel_size={args.data_parallel_size},max_model_len={args.max_model_len},add_bos_token=True,seed=42"
    # Copy the all files from template folder to the work folder
    copy_and_prepare(args)
    # Prepare the datasets for the IFeval and MATH_Hard tasks as we need to join the original dataset
    prepare_datasets(args)
    print(
        f"prepration for the {args.model_name} using {args.evals_dataset} is done, all saved the work_dir: {args.work_dir}"
    )
    command_str = f"lm_eval --model vllm   --model_args {args.model_args} --tasks {args.tasks} --batch_size auto --output_path { args.output_path} --include_path {os.path.abspath(args.work_dir)} --seed 42 "
    if args.limit:
        command_str += f" --limit {args.limit}"
    if args.log_samples:
        command_str += " --log_samples "
    if args.show_config:
        command_str += " --show_config "
    print("please use the following command to run the meta reproduce evals:")
    print(command_str)
