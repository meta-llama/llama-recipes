# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
import glob
import numpy as np
import lm_eval
from lm_eval import tasks
from lm_eval.utils import make_table
from prepare_dataset import get_ifeval_data, get_math_data
import shutil, errno
import yaml
from datetime import datetime

def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def setup_logging(verbosity):
    logging.basicConfig(
        level=verbosity.upper(), format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

def change_yaml(args, base_name):
    for yaml_file in glob.glob(args.template_dir+'**/*/*.yaml', recursive=True):       
        with open(yaml_file, "r") as sources:
            lines = sources.readlines()
        output_path = yaml_file.replace(args.template_dir,args.work_dir)
        print(f"changing {yaml_file} to output_path: {output_path}")
        path = Path(output_path)
        yaml_dir = path.parent
        with open(output_path, "w") as output:
            for line in lines:
                output.write(line.replace("Meta-Llama-3.1-8B",base_name).replace("WORK_DIR",str(yaml_dir)))
def handle_output(args, results, logger):
    if not results:
        logger.error("No results found.")
        sys.exit(1)
    if not args.output_path:
        if args.log_samples:
            logger.error("Specify --output_path for logging samples.")
            sys.exit(1)
        return

    if args.log_samples:
        samples = results.pop("samples")
    results_str = json.dumps(
        results, indent=2, default=_handle_non_serializable, ensure_ascii=False
    )
    if args.show_config:
        logger.info(results_str)
    date_id = datetime.now().isoformat().replace(":", "-")
    path = Path(args.output_path)


    output_dir = path.parent if path.suffix in (".json", ".jsonl") else path
    output_dir.mkdir(parents=True, exist_ok=True)


    file_path = os.path.join(args.output_path, "eval_results_" + date_id + ".json")
    with open(file_path , "w", encoding="utf-8") as f:
        f.write(results_str)

    if args.log_samples:
        for task_name, _ in results.get("configs", {}).items():
            output_name = task_name + "_"+ date_id + re.sub(r"/|=", "_", args.model_args.split(",")[0].replace("pretrained",""))
            sample_file = output_dir.joinpath(f"{output_name}.jsonl")
            sample_data = json.dumps(
                samples.get(task_name, {}), indent=2, default=_handle_non_serializable
            )
            sample_file.write_text(sample_data, encoding="utf-8")

    batch_sizes = ",".join(map(str, results.get("config", {}).get("batch_sizes", [])))
    summary = f"{args.model_name} ({args.model_args})"
    logger.info(summary)
    logger.info(make_table(results))
    if "groups" in results:
        logger.info(make_table(results, "groups"))


def load_tasks(args):
    if not args.tasks or "meta" not in args.tasks:
        raise ValueError("Please specify a valid meta task name")
    if args.tasks:
        tasks_list = args.tasks.split(",") 
    else:
        print("No tasks specified. Please try again")
        sys.exit(1)
    current_dir = os.getcwd()
    config_dir = os.path.join(current_dir, args.work_dir)
    print(f"Including the config_dir to task manager: {config_dir}")
    task_manager = tasks.TaskManager(include_path=config_dir)
    return task_manager, tasks_list

def copy_and_prepare(args):
    if not os.path.exists(args.work_dir):
        # Copy the all files, including yaml files and python files, from template folder to the work folder

        copy_dir(args.template_dir,args.work_dir)
    else:
        print("work_dir already exists, no need to copy files")
    # Use the template yaml to get the correct model name in work_dir yaml
    base_name = args.evals_dataset.split("/")[-1].replace("-evals","").replace("-Instruct","")
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

def prepare_datasets(task_list,args):
    # Prepare the dataset for the IFeval and MATH_Hard tasks as we need to join the original dataset with the evals dataset by the actual questions.
    # model_name are derived from the evals_dataset name
    model_name = args.evals_dataset.split("/")[-1].replace("-evals","")
    if "meta_instruct" in task_list:
        get_ifeval_data(model_name,args.work_dir)
        
        get_math_data(model_name,args.work_dir)
    else:
        if "meta_ifeval" in task_list:
            get_ifeval_data(model_name,args.work_dir)
        if "meta_math_hard" in task_list:
            get_math_data(model_name,args.work_dir)
    
def evaluate_model(args):
    # Customized model such as Quantized model etc.
    # In case you are working with a custom model, you can use the following guide to add it here:
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage
    task_manager, task_list = load_tasks(args)
    logger.info(f"Loaded tasks: {task_list}")
    # We need to prepare the dataset for the IFeval and MATH_Hard tasks
    if "meta_instruct" in task_list or "meta_ifeval" in task_list or "meta_math_hard" in task_list:
        prepare_datasets(task_list, args)
    # Evaluate
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=args.model_args,
        tasks=task_list,
        limit=args.limit,
        log_samples=args.log_samples,
        task_manager=task_manager,
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        fewshot_random_seed=42
        )
    handle_output(args, results, logger)


def copy_dir(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise
def load_config(config_path: str = "./config.yaml"):
    # Read the YAML configuration file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    args = parse_eval_args()
    config = load_config(args.config_path)
    # Create VLLM model args
    for k,v in config.items():
        args.__setattr__(k,v)
    if not os.path.exists(args.template_dir):
        raise ValueError("The template_dir does not exist, please check the path")
    if args.evals_dataset not in ["meta-llama/Meta-Llama-3.1-8B-Instruct-evals","meta-llama/Meta-Llama-3.1-70B-Instruct-evals","meta-llama/Meta-Llama-3.1-405B-Instruct-evals","meta-llama/Meta-Llama-3.1-8B-evals","meta-llama/Meta-Llama-3.1-70B-evals","meta-llama/Meta-Llama-3.1-405B-evals"]:
        raise ValueError("The evals dataset is not valid, please double check the name, must use the name in the Llama 3.1 Evals collection")
    args.model_args = f"pretrained={args.model_name},tensor_parallel_size={args.tensor_parallel_size},dtype=auto,gpu_memory_utilization={args.gpu_memory_utilization},data_parallel_size={args.data_parallel_size},max_model_len={args.max_model_len},add_bos_token=True,seed=42"
    # Copy the all files from template folder to the work folder
    copy_and_prepare(args)
    logger = setup_logging(args.verbosity)
    evaluate_model(args)
