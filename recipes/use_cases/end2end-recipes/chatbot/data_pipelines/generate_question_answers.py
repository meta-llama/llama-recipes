# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import argparse
import asyncio
import json
from config import load_config
from generator_utils import generate_question_batches, parse_qa_to_json, generate_data_eval
from itertools import chain
import logging
import aiofiles  # Ensure aiofiles is installed for async file operations
from abc import ABC, abstractmethod
from octoai.client import OctoAI
from functools import partial
from openai import OpenAI

# Configure logging to include the timestamp, log level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Manage rate limits with throttling
rate_limit_threshold = 2000
allowed_concurrent_requests = int(rate_limit_threshold * 0.75)
request_limiter = asyncio.Semaphore(allowed_concurrent_requests)
# Since OctoAI has different naming for llama models, create this mapping to get huggingface offical model name given OctoAI names.
MODEL_NAME_MAPPING={"meta-llama-3-70b-instruct":"meta-llama/Meta-Llama-3-70B-Instruct",
"meta-llama-3-8b-instruct":"meta-llama/Meta-Llama-3-8B-Instruct","llama-2-7b-chat":"meta-llama/Llama-2-7b-chat-hf"
,"llama-2-70b-chat":"meta-llama/Llama-2-70b-chat-hf"}
class ChatService(ABC):
    @abstractmethod
    async def execute_chat_request_async(self, api_context: dict, chat_request, eval=False):
        pass

# Please implement your own chat service class here.
# The class should inherit from the ChatService class and implement the execute_chat_request_async method.
# The following are two example chat service classes that you can use as a reference.
class OctoAIChatService(ChatService):
    async def execute_chat_request_async(self, api_context: dict, chat_request, eval=False):
        async with request_limiter:
            try:
                event_loop = asyncio.get_running_loop()
                client = OctoAI(api_context['api_key'])
                api_chat_call = partial(
                    client.chat.completions.create,
                    model=api_context['model'],
                    messages=chat_request,
                    temperature=0.0
                )
                response = await event_loop.run_in_executor(None, api_chat_call)
                assistant_response = next((choice.message.content for choice in response.choices if choice.message.role == 'assistant'), "")
                if eval:
                    assistant_response_json = json.loads(assistant_response)
                else:
                    assistant_response_json = parse_qa_to_json(assistant_response)

                return assistant_response_json
            except Exception as error:
                logging.error(f"Error during chat request execution: {error}",exc_info=True)
                return ""
# Use the local vllm openai compatible server for generating question/answer pairs to make API call syntax consistent
# please read for more detail:https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html.
class VllmChatService(ChatService):
    async def execute_chat_request_async(self, api_context: dict, chat_request, eval=False):
        async with request_limiter:
            try:
                event_loop = asyncio.get_running_loop()
                model_name = MODEL_NAME_MAPPING[api_context['model']]
                client = OpenAI(api_key=api_context['api_key'], base_url="http://localhost:"+ str(api_context['endpoint'])+"/v1")
                api_chat_call = partial(
                    client.chat.completions.create,
                    model=model_name,
                    messages=chat_request,
                    temperature=0.0
                )
                response = await event_loop.run_in_executor(None, api_chat_call)
                assistant_response = next((choice.message.content for choice in response.choices if choice.message.role == 'assistant'), "")
                if eval:
                    assistant_response_json = json.loads(assistant_response)
                else:
                    assistant_response_json = parse_qa_to_json(assistant_response)
                return assistant_response_json
            except Exception as error:
                logging.error(f"Error during chat request execution: {error}",exc_info=True)
                return ""

async def main(context):
    if context["endpoint"]:
        chat_service = VllmChatService()
    else:
        chat_service = OctoAIChatService()
    try:
        logging.info("Starting to generate question/answer pairs.")
        data = await generate_question_batches(chat_service, context)
        if not data:
            logging.warning("No data generated. Please check the input context or model configuration.")
            return
        flattened_list = list(chain.from_iterable(data))
        # with open("data.json") as fp:
        #     flattened_list = json.load(fp)
        logging.info(f"Successfully generated {len(flattened_list)} question/answer pairs.")
        # Use asynchronous file operation for writing to the file

        # async with aiofiles.open("data.json", "w") as output_file:
        #     await output_file.write(json.dumps(flattened_list, indent=4))
        # logging.info("Data successfully written to 'data.json'. Process completed.")
        curated_data = await generate_data_eval(chat_service, context,flattened_list)
        logging.info(f"Only {len(curated_data)} question/answer pairs pass the self-curation")
        async with aiofiles.open("curated_data.json", "w") as curated_data:
             await curated_data.write(json.dumps(flattened_list, indent=4))
        logging.info("Data successfully written to 'curated_data.json'. Process completed.")
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
        default="config.yaml",
        help="Set the configuration file path that has system prompt along with language, dataset path and number of questions."
    )
    parser.add_argument(
        "-v", "--vllm_endpoint",
        default=None,
        type=int,
        help="If a port is specified, then use local vllm endpoint for generating question/answer pairs."
    )
    return parser.parse_args()

if __name__ == "__main__":
    logging.info("Initializing the process and loading configuration...")
    args = parse_arguments()

    context = load_config(args.config_path)
    context["total_questions"] = args.total_questions
    context["model"] = args.model
    context["endpoint"] = args.vllm_endpoint
    logging.info(f"Configuration loaded. Generating {args.total_questions} question/answer pairs using model '{args.model}'.")
    if context["endpoint"]:
        logging.info(f"Use local vllm service at port: '{args.vllm_endpoint}'.")
    asyncio.run(main(context))
