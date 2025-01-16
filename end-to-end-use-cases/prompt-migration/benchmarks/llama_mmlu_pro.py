import typing as t

from datasets import load_dataset
import dspy

from .datatypes import TaskDatasets
from .helpers import train_val_test_split


def datasets(
    train_size: float = 0.1,
    validation_size: float = 0.2,
) -> TaskDatasets:
    """
    TODO:
    Load dataset, dataset should be datasets.Dataset type (NOT DatasetDict, OR split the dataset yourself how you want)
    """
    dataset = load_dataset("TODO")
    return train_val_test_split(dataset, _task_doc_example, train_size, validation_size)


class TaskDoc(t.TypedDict):
    problem: str
    gold: str


inputs = ["problem"]
outputs = ["answer"]


def _task_doc_example(doc: TaskDoc) -> dspy.Example:
    return dspy.Example(
        problem=doc["problem"],
        answer=doc["gold"],
    ).with_inputs(*inputs)


def signature(instructions: str = "") -> dspy.Signature:
    class MMLUPro(dspy.Signature):
        __doc__ = instructions
        problem: str = dspy.InputField()
        answer: str = dspy.OutputField()

    return MMLUPro


def metric(gold: dspy.Example, pred: dspy.Example, trace=False) -> bool:
    return gold.answer == pred.answer
