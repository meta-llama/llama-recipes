# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.

import argparse
import asyncio
import json
from config import load_config
from generator_utils import generate_question_batches, generate_data_curation
from chat_utils import OctoAIChatService, VllmChatService
import logging
import aiofiles  # Ensure aiofiles is installed for async file operations


# Configure logging to include the timestamp, log level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def main(context):
    if context["endpoint"]:
        chat_service = VllmChatService()
    else:
        chat_service = OctoAIChatService()
    try:
        logging.info("Starting to generate question/answer pairs.")
        # Generate question/answer pairs as list
        data = await generate_question_batches(chat_service, context)
        if not data:
            logging.warning("No data generated. Please check the input context or model configuration.")
            return
        logging.info(f"Successfully generated {len(data)} question/answer pairs.")
        if context["use_curation"]:
            logging.info("Starting to do self-curation using LLM.")
            data = await generate_data_curation(chat_service, context,data)
            logging.info(f"Only {len(data)} question/answer pairs pass the self-curation")
        async with aiofiles.open(context['output_path'], "w") as output_file:
             await output_file.write(json.dumps(data, indent=4))
        logging.info(f"Data successfully written to {context['output_path']}. Process completed.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the process: {e}",exc_info=True)

def parse_arguments():
    # Define command line arguments for the script
    parser = argparse.ArgumentParser(
        description="Generate question/answer pairs from documentation."
    )
    parser.add_argument(
        "-t", "--total_questions",
        type=int,
        default=100,
        help="Specify the total number of question/answer pairs to generate."
    )
    parser.add_argument(
        "-m", "--model",
        choices=["meta-llama-3-70b-instruct","meta-llama-3-8b-instruct","llama-2-13b-chat", "llama-2-70b-chat"],
        default="meta-llama-3-70b-instruct",
        help="Select the model to use for generation."
    )
    parser.add_argument(
        "-c", "--config_path",
        default="./generation_config.yaml",
        help="Set the configuration file path that has system prompt along with language, dataset path and number of questions."
    )
    parser.add_argument(
        "-v", "--vllm_endpoint",
        default=None,
        type=int,
        help="If a port is specified, then use local vllm endpoint for generating question/answer pairs."
    )
    parser.add_argument(
        "-o", "--output_path",
        default="./data.json",
        help="set the output path for the generated QA pairs. Default is data.json"
    )
    return parser.parse_args()

if __name__ == "__main__":
    logging.info("Initializing the process and loading configuration...")
    args = parse_arguments()

    context = load_config(args.config_path)
    context["total_questions"] = args.total_questions
    context["model"] = args.model
    context["endpoint"] = args.vllm_endpoint
    # If curation prompt is not empty, then use self-curation
    context["use_curation"] = len(context["curation_prompt_template"]) > 0
    context["output_path"] = args.output_path
    logging.info(f"Configuration loaded. Generating {args.total_questions} question/answer pairs using model '{args.model}'.")
    if context["endpoint"]:
        logging.info(f"Use local vllm service at port: '{args.vllm_endpoint}'.")
    asyncio.run(main(context))
