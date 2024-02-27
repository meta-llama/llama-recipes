import argparse
import asyncio
import json
from config import load_config
from generator_utils import generate_question_batches
from itertools import chain
import logging
import aiofiles  # Ensure aiofiles is installed for async file operations

# Configure logging to include the timestamp, log level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main(context):
    try:
        logging.info("Starting to generate question/answer pairs.")
        data = await generate_question_batches(context)
        if not data:
            logging.warning("No data generated. Please check the input context or model configuration.")
            return
        flattened_list = list(chain.from_iterable(data))
        logging.info(f"Successfully generated {len(flattened_list)} question/answer pairs.")
        # Use asynchronous file operation for writing to the file
        async with aiofiles.open("data.json", "w") as output_file:
            await output_file.write(json.dumps(flattened_list, indent=4))
        logging.info("Data successfully written to 'data.json'. Process completed.")

    except Exception as e:
        logging.error(f"An unexpected error occurred during the process: {e}")

def parse_arguments(context):
    # Define command line arguments for the script
    parser = argparse.ArgumentParser(
        description="Generate question/answer pairs from documentation."
    )
    parser.add_argument(
        "-t", "--total_questions",
        type=int,
        default=context["total_questions"],
        help="Specify the number of question/answer pairs to generate."
    )
    parser.add_argument(
        "-m", "--model",
        choices=["gpt-3.5-turbo-16k", "gpt-3.5-turbo-0125"],
        default="gpt-3.5-turbo-16k",
        help="Select the model to use for generation."
    )
    return parser.parse_args()

if __name__ == "__main__":
    logging.info("Initializing the process and loading configuration...")
    context = load_config()
    args = parse_arguments(context)

    context["total_questions"] = args.total_questions
    context["model"] = args.model

    logging.info(f"Configuration loaded. Generating {args.total_questions} question/answer pairs using model '{args.model}'.")
    asyncio.run(main(context))
