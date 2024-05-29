import mdc
from mdc import MDC
import logging
from typing import Literal, Any
from openai import OpenAI
import json
import random
import os, shutil
import argparse
import asyncio
from raft_utils import generate_questions, add_chunk_to_dataset
from chat_utils import OctoAIChatService, VllmChatService
from format import DatasetConverter, datasetFormats, outputDatasetTypes
from config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NUM_DISTRACT_DOCS = 5 # number of distracting documents to add to each chunk
ORCALE_P = 0.8 # probability of related documents to be added to each chunk
async def main(context):
    ds = None
    if context["endpoint"]:
        chat_service = VllmChatService()
    else:
        chat_service = OctoAIChatService()
    try:
        logging.info("Starting to generate question pair.")
        # Generate questions as list for each chunk
        chunk_questions_zip = await generate_questions(chat_service, context)
        if not chunk_questions_zip:
            logging.warning("No questions generated from text. Please check the input context or model configuration.")
            return
        for chunk, questions in chunk_questions_zip:
            logging.info(f"Chunk: {chunk}, question length: {len(questions)}")
            for question in questions:
                logging.info(f"Question: {question}")
        logging.info(f"Successfully generated {sum([len(q) for c,q in chunk_questions_zip])} question/answer pairs.")
        ds = await add_chunk_to_dataset(chunk_questions_zip,context, chat_service,ds,NUM_DISTRACT_DOCS, ORCALE_P)
        print(ds[0])
        ds.save_to_disk(args.output)
        logging.info(f"Data successfully written to {context['output']}. Process completed.")
        formatter = DatasetConverter()

        # Extract format specific params
        format_params = {}
        formatter.convert(ds=ds, format=args.output_format, output_path=args.output, output_type=args.output_type, params=format_params)
    except Exception as e:
        logging.error(f"An unexpected error occurred during the process: {e}",exc_info=True)

def parse_arguments():
    # Define command line arguments for the script
    parser = argparse.ArgumentParser(
        description="Generate question/answer pairs from documentation."
    )
    parser.add_argument(
        "-t", "--questions_per_chunk",
        type=int,
        default=3,
        help="Specify the number of question pairs to generate per chunk."
    )
    parser.add_argument(
        "-m", "--model",
        choices=["meta-llama-3-70b-instruct","meta-llama-3-8b-instruct","llama-2-13b-chat", "llama-2-70b-chat"],
        default="meta-llama-3-70b-instruct",
        help="Select the model to use for generation."
    )
    parser.add_argument(
        "-c", "--config_path",
        default="./raft.yaml",
        help="Set the configuration file path that has system prompt along with language, dataset path and number of questions."
    )
    parser.add_argument(
        "-v", "--vllm_endpoint",
        default=None,
        type=int,
        help="If a port is specified, then use local vllm endpoint for generating question/answer pairs."
    )
    parser.add_argument("--chunk_size", type=int, default=512, help="The size of each chunk in number of tokens")
    parser.add_argument("-o","--output", type=str, default="./", help="The path at which to save the dataset")
    parser.add_argument("--output-format", type=str, default="hf", help="Format to convert the dataset to. Defaults to hf.", choices=datasetFormats)
    parser.add_argument("--output-type", type=str, default="jsonl", help="Type to export the dataset to. Defaults to jsonl.", choices=outputDatasetTypes)
    return parser.parse_args()

if __name__ == "__main__":
    logging.info("Initializing the process and loading configuration...")
    args = parse_arguments()

    context = load_config(args.config_path)
    context["questions_per_chunk"] = args.questions_per_chunk
    context["model"] = args.model
    context["chunk_size"] = args.chunk_size
    context["endpoint"] = args.vllm_endpoint
    context["output"] = args.output
    logging.info(f"Configuration loaded. Generating {args.questions_per_chunk} question per chunk using model '{args.model}'.")
    if context["endpoint"]:
        logging.info(f"Use local vllm service at port: '{args.vllm_endpoint}'.")
    asyncio.run(main(context))
