import json
import re
import typing as t

import dspy

from datasets import load_dataset

from .datatypes import TaskDatasets
from .helpers import train_val_test_split


def datasets(
    train_size: float = 0.1,
    validation_size: float = 0.2,
) -> TaskDatasets:
    """
    Load dataset, dataset should be datasets.Dataset type (NOT DatasetDict, OR split the dataset yourself how you want)
    """
    dataset = load_dataset(
        "meta-llama/Llama-3.3-70B-Instruct-evals",
        "Llama-3.3-70B-Instruct-evals__mmlu_pro__details",
    )
    return train_val_test_split(
        dataset["latest"],
        _task_doc_example,
        train_size,
        validation_size,
    )


class TaskDoc(t.TypedDict):
    task_type: str
    task_name: str
    subtask_name: str
    input_question: str
    input_choice_list: dict
    input_final_prompts: list
    input_correct_responses: list
    output_prediction_text: list
    output_parsed_answer: str
    output_choice_completions: t.Optional[dict]
    output_choice_negative_log_likelihoods: t.Optional[dict]
    output_metrics: dict
    is_correct: bool
    input_question_hash: str
    input_final_prompts_hash: list
    benchmark_label: str
    eval_config: dict


inputs = ["input_question", "input_choice_list"]
outputs = ["output_parsed_answer"]


class CustomJSONAdapter(dspy.JSONAdapter):
    def parse(self, signature, completion):
        try:
            try:
                fields = json.loads(completion)
            except:
                fields = {"reasoning": completion, "answer": ""}

            if isinstance(fields, list):
                fields = fields[0] if fields else {"reasoning": "", "answer": ""}

            if "reasoning" not in fields:
                fields["reasoning"] = ""
            if "answer" not in fields:
                reasoning = fields.get("reasoning", "")
                match = re.search(
                    r"\b([A-J])\b|answer\s+is\s+([A-J])\b", reasoning, re.IGNORECASE
                )
                fields["answer"] = (
                    (match.group(1) or match.group(2)).upper() if match else ""
                )

            return fields
        except Exception as e:
            return {"reasoning": "", "answer": ""}


def signature(instructions: str = "") -> dspy.Signature:
    """Define the signature for MMLU Pro task."""

    class MMLUPro(dspy.Signature):
        """Multiple choice question answering with reasoning."""

        question: str = dspy.InputField(desc="The question to be answered")
        options: dict = dspy.InputField(desc="Dictionary of answer choices")
        reasoning: str = dspy.OutputField(
            desc="Step by step reasoning to arrive at the answer"
        )
        answer: str = dspy.OutputField(desc="The correct answer letter (A-J)")

    dspy.settings.configure(adapter=CustomJSONAdapter())

    return MMLUPro


def _task_doc_example(doc: TaskDoc) -> dspy.Example:
    """Create an example with proper input/output key configuration."""
    example = dspy.Example(
        question=doc["input_question"],
        options=doc["input_choice_list"],
        reasoning="",  # Initialize empty reasoning
        answer=doc["output_parsed_answer"] if doc["output_parsed_answer"] else "",
    )
    example._input_keys = {"question", "options"}
    example._output_keys = {"reasoning", "answer"}
    return example


def metric(gold: dspy.Example, pred: dspy.Example, trace=False) -> bool:
    """
    Compares gold and predicted answers while handling various response formats.
    Ensures answer field is always present by extracting from reasoning if needed.
    """
    try:
        pred_dict = pred if isinstance(pred, dict) else pred.__dict__

        reasoning = pred_dict.get("reasoning", "")
        if isinstance(reasoning, str) and "answer" not in pred_dict:
            match = re.search(
                r"\b([A-J])\b|answer\s+is\s+([A-J])\b", reasoning, re.IGNORECASE
            )
            if match:
                answer = match.group(1) or match.group(2)
                pred_dict["answer"] = answer.upper()

        pred_answer = pred_dict.get("answer", "")
        if isinstance(pred_answer, str):
            pred_answer = pred_answer.strip().upper()
            if len(pred_answer) > 1:
                pred_answer = pred_answer[0]

        gold_answer = gold.answer if hasattr(gold, "answer") else ""
        if isinstance(gold_answer, str):
            gold_answer = gold_answer.strip().upper()

        # Handle empty answers
        if not gold_answer or not pred_answer:
            return False

        return gold_answer == pred_answer

    except Exception as e:
        if trace:
            print(f"Error in metric: {str(e)}")
        return False
