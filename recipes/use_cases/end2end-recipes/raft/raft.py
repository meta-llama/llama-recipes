import mdc
from mdc import MDC
import logging
from typing import Literal, Any
from openai import OpenAI
import datasets
from datasets import Dataset, load_dataset
import json
import random
import os, shutil
import argparse
import asyncio
from raft_utils import generate_questions, add_chunk_to_dataset
from chat_utils import OctoAIChatService, VllmChatService
from format import DatasetConverter, datasetFormats, outputDatasetTypes
from config import load_config

# def generate_label(client: OpenAI, question: str, context: Any, doctype: DocType = "pdf", model: str = None) -> str | None:
#     """
#     Generates the label / answer to `question` using `context` and GPT-4.
#     """
#     question = encode_question(question, context) if doctype == "api" else encode_question_gen(question, context)
#     response = client.chat.completions.create(
#         model=model,
#         messages=question,
#         n=1,
#         temperature=0
#     )
#     response = response.choices[0].message.content
#     return response
# Configure logging to include the timestamp, log level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def main(context):
    if context["endpoint"]:
        chat_service = VllmChatService()
    else:
        chat_service = OctoAIChatService()
    try:
        logging.info("Starting to generate question pair.")
        # Generate question/answer pairs as list
        chunks = await generate_questions(chat_service, context)
        if not chunks:
            logging.warning("No questions generated from text. Please check the input context or model configuration.")
            return
        logging.info(f"Successfully generated {sum([len(q) for q in chunks])} question/answer pairs.")
        print(chunks)
        for i, chunk in enumerate(chunks):
            perc = ceil(i / num_chunks * 100)
            with MDC(progress=f"{perc}%"):
                logger.info(f"Adding chunk {i}/{num_chunks}")
                add_chunk_to_dataset(client, chunks, chunk, args.doctype, args.questions, NUM_DISTRACT_DOCS, model=args.completion_model)

        logging.info(f"Data successfully written to {context['output']}. Process completed.")
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
