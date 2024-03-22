# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import lm_eval
from lm_eval import evaluator, tasks
from lm_eval.utils import make_table


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


def handle_output(args, results, logger):
    if not args.output_path:
        if args.log_samples:
            logger.error("Specify --output_path for logging samples.")
            sys.exit(1)
        logger.info(json.dumps(results, indent=2, default=_handle_non_serializable))
        return

    path = Path(args.output_path)
    if path.is_file() or path.with_name("results.json").is_file():
        logger.warning(f"File already exists at {path}. Results will be overwritten.")

    output_dir = path.parent if path.suffix in (".json", ".jsonl") else path
    output_dir.mkdir(parents=True, exist_ok=True)

    results_str = json.dumps(results, indent=2, default=_handle_non_serializable)
    if args.show_config:
        logger.info(results_str)

    file_path = os.path.join(args.output_path, "results.json")
    with open(file_path , "w", encoding="utf-8") as f:
        f.write(results_str)

    if args.log_samples:
        samples = results.pop("samples", {})
        for task_name, _ in results.get("configs", {}).items():
            output_name = re.sub(r"/|=", "__", args.model_args) + "_" + task_name
            sample_file = output_dir.joinpath(f"{output_name}.jsonl")
            sample_data = json.dumps(
                samples.get(task_name, {}), indent=2, default=_handle_non_serializable
            )
            sample_file.write_text(sample_data, encoding="utf-8")

    batch_sizes = ",".join(map(str, results.get("config", {}).get("batch_sizes", [])))
    summary = f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    logger.info(summary)
    logger.info(make_table(results))
    if "groups" in results:
        logger.info(make_table(results, "groups"))


def load_tasks(args):
    tasks.initialize_tasks()
    if args.open_llm_leaderboard_tasks:
        current_dir = os.getcwd()
        config_dir = os.path.join(current_dir, "open_llm_leaderboard")
        lm_eval.tasks.include_path(config_dir)
        return [
            "arc_challenge_25_shot",
            "hellaswag_10_shot",
            "truthfulqa_mc2",
            "winogrande_5_shot",
            "gsm8k",
            "mmlu",
        ]
    return args.tasks.split(",") if args.tasks else []


def parse_eval_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model", "-m", default="hf", help="Name of model, e.g., `hf`."
    )
    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        help="Comma-separated list of tasks, or 'list' to display available tasks.",
    )
    parser.add_argument(
        "--model_args",
        "-a",
        default="",
        help="Comma-separated string arguments for model, e.g., `pretrained=EleutherAI/pythia-160m`.",
    )
    parser.add_argument(
        "--open_llm_leaderboard_tasks",
        "-oplm",
        action="store_true",
        default=False,
        help="Choose the list of tasks with specification in HF open LLM-leaderboard.",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        help="Number of examples in few-shot context.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        default=1,
        help="Batch size, can be 'auto', 'auto:N', or an integer.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size with 'auto' batch size.",
    )
    parser.add_argument(
        "--device", default=None, help="Device for evaluation, e.g., 'cuda', 'cpu'."
    )
    parser.add_argument(
        "--output_path", "-o", type=str, default=None, help="Path for saving results."
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        help="Limit number of examples per task.",
    )
    parser.add_argument(
        "--use_cache", "-c", default=None, help="Path to cache db file, if used."
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        default="INFO",
        help="Logging level: CRITICAL, ERROR, WARNING, INFO, DEBUG.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default=None,
        help="Generation kwargs for tasks that support it.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks.",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to include if there are external tasks.",
    )
    parser.add_argument(
        "--decontamination_ngrams_path", default=None
    )  # Not currently used
    return parser.parse_args()


def evaluate_model(args):
    try:
        task_list = load_tasks(args)
        # Customized model such as Quantized model etc.
        # In case you are working with a custom model, you can use the following guide to add it here:
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage

        # Evaluate
        results = evaluator.simple_evaluate(
            model=args.model,
            model_args=args.model_args,
            tasks=task_list,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=args.device,
            use_cache=args.use_cache,
            limit=args.limit,
            decontamination_ngrams_path=args.decontamination_ngrams_path,
            check_integrity=args.check_integrity,
            write_out=args.write_out,
            log_samples=args.log_samples,
            gen_kwargs=args.gen_kwargs,
        )
        handle_output(args, results, logger)

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_eval_args()
    logger = setup_logging(args.verbosity)
    evaluate_model(args)
