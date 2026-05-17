"""
Utility functions for structured data extraction.

This module provides helper functions for working with JSON schemas, encoding images,
extracting structured data from LLM responses, and logging.
"""

import ast
import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pymupdf
import yaml
from openai import OpenAI

from typedicts import InferenceRequest, VLLMInferenceRequest

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


def setup_logger(logfile, verbose=False):
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the file handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # If verbose, also add a console handler
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = logging.getLogger(__name__)

# Compile regex patterns once for better performance
JSON_BLOCK_OPEN = re.compile(r"```json")
JSON_BLOCK_CLOSE = re.compile(r"}\s+```")


# Configuration management
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file. If None, uses default path.

    Returns:
        Dict containing configuration values

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the configuration file is invalid
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in configuration file: {e}")
        raise


# Load configuration
config = load_config()


# LLM Singleton
class LLMSingleton:
    """Singleton class for managing LLM instances."""

    _instance = None

    @classmethod
    def get_instance(cls) -> LLM:
        """
        Get or create the LLM instance.

        Returns:
            LLM: An initialized VLLM model instance
        """
        if cls._instance is None:
            try:
                cls._instance = LLM(
                    config["model"]["path"],
                    tensor_parallel_size=config["model"]["tensor_parallel_size"],
                    max_model_len=config["model"]["max_model_len"],
                    max_num_seqs=config["model"]["max_num_seqs"],
                )
                logger.info(f"Initialized LLM with model: {config['model']['path']}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                raise
        return cls._instance


class ImageUtils:
    """Utility functions for working with images."""

    @staticmethod
    def encode_image(image_path: Union[Path, str]) -> str:
        """
        Encode an image to base64.

        Args:
            image_path: Path to the image file

        Returns:
            Base64-encoded string representation of the image

        Raises:
            FileNotFoundError: If the image file doesn't exist
        """
        if isinstance(image_path, str):
            image_path = Path(image_path)
        try:
            return base64.b64encode(image_path.read_bytes()).decode("utf-8")
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            raise


class JSONUtils:
    """Utility functions for working with JSON data."""

    @staticmethod
    def extract_json_blocks(content: str) -> List[str]:
        """
        Extract JSON code blocks from markdown-formatted text.

        Parses a string containing markdown-formatted text and extracts all JSON blocks
        that are enclosed in ```json ... ``` code blocks. This is useful for extracting
        structured data from LLM responses.

        Args:
            content: The markdown-formatted text containing JSON code blocks

        Returns:
            List[str]: A list of extracted JSON strings (without the markdown delimiters)
        """
        blocs_ix = []
        str_ptr = 0

        while str_ptr < len(content):
            start_ix = content.find("```json", str_ptr)
            if start_ix == -1:
                break
            start_ix += len("```json")
            end_match = JSON_BLOCK_CLOSE.search(content[start_ix:])
            if end_match:
                end_ix = start_ix + end_match.start() + 1
            else:
                end_ix = len(content)  # no closing tag, take the rest of the string
            blocs_ix.append((start_ix, end_ix))
            str_ptr = end_ix + 1

        return [content[ix[0] : ix[1]].strip() for ix in blocs_ix]

    @staticmethod
    def load_json_from_str(json_str: str) -> Dict[str, Any]:
        """
        Parse a JSON string into a Python dictionary.

        Attempts to parse a string as JSON using multiple methods. First tries standard
        json.loads(), then falls back to ast.literal_eval() if that fails. This provides
        more robust JSON parsing for LLM outputs that might not be perfectly formatted.

        Args:
            json_str: The JSON string to parse

        Returns:
            Dict[str, Any]: The parsed JSON as a dictionary

        Raises:
            ValueError: If parsing fails
        """
        if not isinstance(json_str, str):
            return json_str

        try:
            return json.loads(json_str)
        except json.decoder.JSONDecodeError:
            # Try with None replacement
            json_str = json_str.replace("null", "None")
            try:
                return ast.literal_eval(json_str)
            except:
                raise ValueError(f"Failed to load valid JSON from string: {json_str}")

    @staticmethod
    def extract_json_from_response(content: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from an LLM response.

        Processes a response from an LLM that may contain JSON in a markdown code block.
        First checks if the response contains markdown-formatted JSON blocks and extracts them,
        then parses the JSON string into a Python dictionary.

        Args:
            content: The LLM response text that may contain JSON

        Returns:
            Dict[str, Any]: The parsed JSON as a dictionary

        Raises:
            ValueError: If extraction or parsing fails
        """
        try:
            if "```json" in content:
                json_blocks = JSONUtils.extract_json_blocks(content)
                if not json_blocks:
                    raise ValueError("No JSON blocks found in response")
                content = json_blocks[-1]

            return JSONUtils.load_json_from_str(content)
        except Exception as e:
            raise ValueError(f"Failed to extract JSON from response: {str(e)}")

    @staticmethod
    def make_all_fields_required(schema: Dict[str, Any]) -> None:
        """
        Make all fields in a JSON schema required.

        Recursively modifies the JSON schema in-place, so that every property in each 'properties'
        is added to the 'required' list at that schema level. This ensures that the LLM will
        attempt to extract all fields defined in the schema.

        Args:
            schema: The JSON schema to modify
        """

        def _process_schema_node(subschema):
            """Process a single node in the schema."""
            if not isinstance(subschema, dict):
                return

            schema_type = subschema.get("type")
            if schema_type == "object" or (
                isinstance(schema_type, list) and "object" in schema_type
            ):
                props = subschema.get("properties")
                if isinstance(props, dict):
                    subschema["required"] = list(props.keys())

            # Recurse into sub-schemas
            for key in ("properties", "definitions", "patternProperties"):
                children = subschema.get(key)
                if isinstance(children, dict):
                    for v in children.values():
                        _process_schema_node(v)

            # Recurse into schema arrays
            for key in ("allOf", "anyOf", "oneOf"):
                children = subschema.get(key)
                if isinstance(children, list):
                    for v in children:
                        _process_schema_node(v)

            # 'items' can be a schema or list of schemas
            items = subschema.get("items")
            if isinstance(items, dict):
                _process_schema_node(items)
            elif isinstance(items, list):
                for v in items:
                    _process_schema_node(v)

            # Extras: 'not', 'if', 'then', 'else'
            for key in ["not", "if", "then", "else"]:
                sub = subschema.get(key)
                if isinstance(sub, dict):
                    _process_schema_node(sub)

        _process_schema_node(schema)


class PDFUtils:
    """Utility functions for working with PDF files."""

    @staticmethod
    def extract_pages(
        pdf_path: Union[str, Path], output_dir: Union[str, Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract pages from a PDF file as images to disk.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images (defaults to /tmp/pdf_images)

        Returns:
            List of dictionaries containing doc_path, image_path, and page_num

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
        """
        if isinstance(pdf_path, str):
            pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        stem = pdf_path.stem
        if output_dir is None:
            output_dir = Path("/tmp/pdf_images")
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True, parents=True)
        pages = []

        try:
            pdf_document = pymupdf.open(pdf_path)
            for page_num, page in enumerate(pdf_document):
                image_path = output_dir / f"{stem}_{page_num}.png"
                pix = page.get_pixmap(dpi=100)
                pix.save(str(image_path))

                pages.append(
                    {
                        "doc_path": str(pdf_path),
                        "image_path": str(image_path),
                        "page_num": page_num,
                    }
                )
            return pages
        except Exception as e:
            logger.error(f"Failed to extract pages from PDF: {e}")
            raise


class InferenceUtils:
    """Utility functions for running inference with LLMs."""

    @staticmethod
    def get_offline_llm() -> LLM:
        """
        Initialize and return a local LLM instance using the singleton pattern.

        Returns:
            LLM: An initialized VLLM model instance
        """
        return LLMSingleton.get_instance()

    @staticmethod
    def make_vllm_batch(
        request_params_batch: Union[InferenceRequest, List[InferenceRequest]],
    ) -> VLLMInferenceRequest:
        """
        Convert one or more inference requests to VLLM batch format.

        Args:
            request_params_batch: Single request parameters or a list of request parameters

        Returns:
            VLLMInferenceRequest: Formatted request for VLLM
        """
        if isinstance(request_params_batch, dict):
            request_params_batch = [request_params_batch]

        sampling_params = []
        messages = []
        for req in request_params_batch:
            params = {
                "top_p": req["top_p"],
                "temperature": req["temperature"],
                "max_tokens": req["max_completion_tokens"],
                "seed": req["seed"],
            }
            if "response_format" in req:
                gd_params = GuidedDecodingParams(
                    json=req["response_format"]["json_schema"]["schema"]
                )
                sampling_params.append(
                    SamplingParams(guided_decoding=gd_params, **params)
                )
            else:
                sampling_params.append(SamplingParams(**params))
            messages.append(req["messages"])

        return {"messages": messages, "sampling_params": sampling_params}

    @staticmethod
    def run_vllm_inference(
        vllm_request: VLLMInferenceRequest,
    ) -> List[str]:
        """
        Run inference on a batch of requests using the local LLM.

        This function processes one or more requests through the local LLM,
        handling the conversion to VLLM format and extracting the raw text
        responses.

        Args:
            vllm_request: Formatted request for VLLM

        Returns:
            List[str]: Raw text responses from the LLM for each request in the batch
        """
        try:
            local_llm = InferenceUtils.get_offline_llm()
            out = local_llm.chat(
                vllm_request["messages"], vllm_request["sampling_params"], use_tqdm=True
            )
            raw_responses = [r.outputs[0].text for r in out]
            return raw_responses
        except Exception as e:
            logger.error(f"VLLM inference failed: {e}")
            raise

    @staticmethod
    def run_openai_inference(request: InferenceRequest) -> str:
        """
        Run inference using OpenAI-compatible API.

        Args:
            request: Inference request parameters

        Returns:
            str: Model response text
        """
        try:
            client = OpenAI(
                base_url=config["model"]["base_url"], api_key=config["model"]["api_key"]
            )
            model_id = config["model"]["model_id"] or client.models.list().data[0].id
            r = client.chat.completions.create(model=model_id, **request)
            if not r.choices or r.choices[0].message is None:
                raise ValueError("No choices returned from LLM")
            return r.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI inference failed: {e}")
            raise

    @staticmethod
    def request_builder(
        user_prompt: str,
        system_prompt: str = None,
        img_path: str = None,
        use_json_decoding: bool = False,
        output_schema: Dict[str, Any] = None,
        **kwargs,
    ) -> InferenceRequest:
        request = kwargs

        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})

        user_content = []
        if img_path:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            img_b64 = ImageUtils.encode_image(img_path)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                }
            )
        user_content.append({"type": "text", "text": user_prompt})
        msgs.append({"role": "user", "content": user_content})
        request["messages"] = msgs

        if use_json_decoding:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "OutputSchema", "schema": output_schema},
            }

        return request


def export_csvs_to_excel_tabs(csv_folder_path, output_excel_path):
    """
    Exports multiple CSV files from a specified folder into a single Excel
    workbook, with each CSV appearing as a separate tab (sheet).

    Args:
        csv_folder_path (str): The path to the folder containing the CSV files.
        output_excel_path (str): The desired path for the output Excel file.
    """
    try:
        # Create an ExcelWriter object
        with pd.ExcelWriter(output_excel_path, engine="xlsxwriter") as writer:
            # Iterate through all files in the specified folder
            for filename in os.listdir(csv_folder_path):
                if filename.endswith(".csv"):
                    csv_file_path = os.path.join(csv_folder_path, filename)
                    sheet_name = os.path.splitext(filename)[0][:31]

                    # Read the CSV file into a pandas DataFrame
                    df = pd.read_csv(csv_file_path)

                    # Write the DataFrame to a new sheet in the Excel file
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Successfully exported CSV files to '{output_excel_path}'")

    except Exception as e:
        print(f"An error occurred: {e}")
