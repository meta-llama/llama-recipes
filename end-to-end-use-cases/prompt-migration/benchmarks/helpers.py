import typing as t

from .datatypes import TaskDatasets

if t.TYPE_CHECKING:
    import dspy
    from datasets import Dataset


def train_val_test_split(
    dataset: "Dataset",
    mapper: t.Callable[[dict], "dspy.Example"],
    train_size: float = 0.1,
    validation_size: float = 0.1,
) -> TaskDatasets:
    docs = dataset.train_test_split(train_size=train_size)
    train_docs = docs["train"]
    docs = docs["test"].train_test_split(train_size=validation_size)
    validation_docs = docs["train"]
    test_docs = docs["test"]
    return TaskDatasets(
        trainset=list(map(mapper, train_docs)),
        valset=list(map(mapper, validation_docs)),
        testset=list(map(mapper, test_docs)),
    )


def fixed_split(
    dataset: "Dataset",
    mapper: t.Callable[[dict], "dspy.Example"],
    train_size: int = 1000,
    validation_size: int = 200,
) -> TaskDatasets:
    """Split dataset by taking first N examples instead of random sampling.

    Args:
        dataset: Input dataset
        mapper: Function to map dataset examples to dspy.Example
        train_size: Number of examples to use for training (default: 1000)
        validation_size: Number of examples to use for validation (default: 200)

    Returns:
        TaskDatasets containing train, validation and test splits
    """
    train_docs = dataset.select(range(train_size))
    validation_docs = dataset.select(range(train_size, train_size + validation_size))
    test_docs = dataset.select(range(train_size + validation_size, len(dataset)))

    return TaskDatasets(
        trainset=list(map(mapper, train_docs)),
        valset=list(map(mapper, validation_docs)),
        testset=list(map(mapper, test_docs)),
    )
