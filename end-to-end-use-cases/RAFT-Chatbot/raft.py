import logging
import os
import argparse
from raft_utils import generate_questions, add_chunk_to_dataset
from format import DatasetConverter, datasetFormats, outputDatasetTypes
from config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(api_config):
    ds = None
    try:
        logging.info("Starting to generate question pair.")
        # Generate questions as list for each chunk
        chunk_questions_zip = generate_questions(api_config)
        if not chunk_questions_zip:
            logging.warning("No questions generated from text. Please check the api_config or model configuration.")
            return
        logging.info(f"Successfully generated {sum([len(q) for c,q in chunk_questions_zip])} question/answer pairs.")
        ds = add_chunk_to_dataset(chunk_questions_zip,api_config)
        ds.save_to_disk(args.output)
        logging.info(f"Data successfully written to {api_config['output']}. Process completed.")
        formatter = DatasetConverter()

        # Extract format specific params
        format_params = {}
        formatter.convert(ds=ds, format=args.output_format, output_path=args.output+"raft", output_type=args.output_type, params=format_params)
    except Exception as e:
        logging.error(f"An unexpected error occurred during the process: {e}",exc_info=True)

def parse_arguments():
    # Define command line arguments for the script
    parser = argparse.ArgumentParser(
        description="Generate RAFT question/answer/context pairs from documentation."
    )
    parser.add_argument(
        "-t", "--questions_per_chunk",
        type=int,
        default=4,
        help="Specify the number of question pairs to generate per chunk."
    )
    parser.add_argument(
        "-m", "--model",
        default="meta-llama/Meta-Llama-3-70B-Instruct",
        help="Select the model to use for generation."
    )
    parser.add_argument(
        "-c", "--config_path",
        default="./raft.yaml",
        help="Set the configuration file path that has system prompt along with language, dataset path and number of questions."
    )
    parser.add_argument(
        "-u", "--endpoint_url",
        default="http://localhost:8001/v1",
        type=str,
        help="LLM API url for generating question/answer pairs."
    )
    parser.add_argument(
        "-k", "--api_key",
        default="EMPTY",
        type=str,
        help="LLM API key for generating question/answer pairs."
    )
    parser.add_argument("--chunk_size", type=int, default=1000, help="The size of each chunk in number of tokens")
    parser.add_argument("-o","--output", type=str, default="./output/", help="The path at which to save the dataset")
    parser.add_argument("--output-format", type=str, default="hf", help="Format to convert the dataset to. Defaults to hf.", choices=datasetFormats)
    parser.add_argument("--output-type", type=str, default="jsonl", help="Type to export the dataset to. Defaults to jsonl.", choices=outputDatasetTypes)
    return parser.parse_args()

if __name__ == "__main__":
    logging.info("Initializing the process and loading configuration...")
    args = parse_arguments()

    api_config = load_config(args.config_path)
    api_config["questions_per_chunk"] = args.questions_per_chunk
    api_config["model"] = args.model
    api_config["chunk_size"] = args.chunk_size
    api_config["endpoint_url"] = args.endpoint_url
    api_config["output"] = args.output
    api_config["api_key"] = args.api_key
    # if OPENAI_API_KEY is defined in the system environment, use it as the API key
    if os.environ.get('API_KEY') is not None:
        api_config["api_key"] = os.environ["API_KEY"]
    logging.info(f"Configuration loaded. Generating {args.questions_per_chunk} question per chunk using model '{args.model}'.")
    logging.info(f"Chunk size: {args.chunk_size}.")
    logging.info(f"num_distract_docs: {api_config['num_distract_docs']}, refusal_probability: {api_config['refusal_probability']}")
    logging.info(f"Will use endpoint_url: {args.endpoint_url}.")
    logging.info(f"Output will be written to {args.output}.")
    main(api_config)
