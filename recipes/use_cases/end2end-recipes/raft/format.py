# file copied from https://github.com/ShishirPatil/gorilla/blob/main/raft/format.py
from abc import ABC, abstractmethod
import argparse
from datasets import Dataset, load_dataset
from typing import Dict, Literal, Any, get_args

"""
This file allows to convert raw HuggingFace Datasets into files suitable to fine tune completion and chat models.
"""

OutputDatasetType = Literal["parquet", "jsonl"]
outputDatasetTypes = list(get_args(OutputDatasetType))

InputDatasetType = Literal["arrow", "jsonl"]
inputDatasetTypes = list(get_args(InputDatasetType))

DatasetFormat = Literal["hf", "completion", "chat"]
datasetFormats = list(get_args(DatasetFormat))

def get_args() -> argparse.Namespace:
    """
    Parses and returns the arguments specified by the user's command
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True, help="Input HuggingFace dataset file")
    parser.add_argument("--input-type", type=str, default="arrow", help="Format of the input dataset. Defaults to arrow.", choices=inputDatasetTypes)
    parser.add_argument("--output", type=str, required=True, help="Output file")
    parser.add_argument("--output-format", type=str, required=True, help="Format to convert the dataset to", choices=datasetFormats)
    parser.add_argument("--output-type", type=str, default="jsonl", help="Type to export the dataset to. Defaults to jsonl.", choices=outputDatasetTypes)
    parser.add_argument("--output-chat-system-prompt", type=str, help="The system prompt to use when the output format is chat")

    args = parser.parse_args()
    return args

class DatasetFormatter(ABC):
    """
    Base class for dataset formatters. Formatters rename columns, remove and add 
    columns to match the expected target format structure. HF, Chat or Completion models file formats.
    https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
    """
    @abstractmethod
    def format(self, ds: Dataset, params: Dict[str, str]) -> Dataset:
        pass

class DatasetExporter(ABC):
    """
    Base class for dataset exporters. Exporters export dataset to different file types, JSONL, Parquet, ...
    """
    @abstractmethod
    def export(self, ds: Dataset, output_path: str):
        pass

class DatasetConverter():
    """
    Entry point class. It resolves which DatasetFormatter and which DatasetExporter to use and runs them.
    """
    formats: Dict[DatasetFormat, DatasetFormatter]
    exporters: Dict[OutputDatasetType, Any]

    def __init__(self) -> None:
        self.formats = {
            "hf": HuggingFaceDatasetFormatter(),
            "completion": OpenAiCompletionDatasetFormatter(),
            "chat": OpenAiChatDatasetFormatter()
        }
        self.exporters = {
            "parquet": ParquetDatasetExporter(),
            "jsonl": JsonlDatasetExporter()
        }

    def convert(self, ds: Dataset, format: DatasetFormat, output_path: str, output_type: OutputDatasetType, params: Dict[str, str]):
        if not format in self.formats:
            raise Exception(f"Output Format {format} is not supported, pleased select one of {self.formats.keys()}")
        
        if not output_type in self.exporters:
            raise Exception(f"Output Type {output_type} is not supported, pleased select one of {self.exporters.keys()}")

        formatter = self.formats[format]
        newds = formatter.format(ds, params)
        exporter = self.exporters[output_type]
        exporter.export(newds, output_path)

class HuggingFaceDatasetFormatter(DatasetFormatter):
    """
    Returns the HuggingFace Dataset as is
    """
    def format(self, ds: Dataset, params: Dict[str, str]) -> Dataset:
        return ds

def _remove_all_columns_but(ds: Dataset, keep_columns) -> Dataset:
    """
    HF Dataset doesn't have a way to copy only specific columns of a Dataset so this help
    removes all columns but the ones specified.
    """
    remove_columns = list(ds.column_names)
    for keep in keep_columns:
        remove_columns.remove(keep)
    ds = ds.remove_columns(remove_columns)
    return ds

class OpenAiCompletionDatasetFormatter(DatasetFormatter):
    """
    Returns the Dataset in the OpenAI Completion Fine-tuning file format with two fields "prompt" and "completion".
    https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
    """
    def format(self, ds: Dataset, params: Dict[str, str]) -> Dataset:
        newds = ds.rename_columns({'question': 'prompt', 'cot_answer': 'completion'})
        return _remove_all_columns_but(newds, ['prompt', 'completion'])

class OpenAiChatDatasetFormatter(OpenAiCompletionDatasetFormatter):
    """
    Returns the Dataset in the OpenAI Chat Fine-tuning file format with one field "messages".
    https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
    """
    def format(self, ds: Dataset, params: Dict[str, str]) -> Dataset:
        newds = super().format(ds, params)

        def format_messages(row):
            messages = []
            if 'system_prompt' in params:
                system_prompt = params['system_prompt']
                messages.append({ "role": "system", "content": system_prompt})
            messages.extend([{ "role": "user", "content": row['prompt']}, { "role": "assistant", "content": row['completion']}])
            chat_row = {"messages": messages}
            return chat_row

        newds = newds.map(format_messages)
        return _remove_all_columns_but(newds, ['messages'])

def append_extension(path: str, extension: str) -> str:
    suffix = "." + extension
    if not path.endswith(suffix):
        path = path + suffix
    return path


class JsonlDatasetExporter(DatasetExporter):
    """
    Exports the Dataset to a JSONL file
    """

    def export(self, ds: Dataset, output_path: str):
        ds.to_json(append_extension(output_path, "jsonl"))


class ParquetDatasetExporter(DatasetExporter):
    """
    Exports the Dataset to a Parquet file
    """

    def export(self, ds: Dataset, output_path: str):
        ds.to_parquet(append_extension(output_path, "parquet"))


def main():
    """
    When raft.py is executed from the command line.
    """
    args = get_args()
    ds = load_dataset(args.input_type, data_files={"train": args.input})['train']
    formatter = DatasetConverter()

    if args.output_chat_system_prompt and args.output_format != "chat":
        raise Exception("Parameter --output-chat-system-prompt can only be used with --output-format chat")

    format_params = {}
    if args.output_chat_system_prompt:
        format_params['system_prompt'] = args.output_chat_system_prompt

    formatter.convert(ds=ds, format=args.output_format, output_path=args.output, output_type=args.output_type, params=format_params)

if __name__ == "__main__":
    main()
