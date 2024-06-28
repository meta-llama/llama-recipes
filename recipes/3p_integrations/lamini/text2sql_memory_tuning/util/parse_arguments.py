from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()

    # The max number of examples to evaluate
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="The max number of examples to evaluate",
        required=False,
    )

    parser.add_argument(
        "--sql-model-name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="The model to use for text2sql",
        required=False,
    )

    parser.add_argument(
        "--gold-file-name",
        type=str,
        default="gold-test-set.jsonl",
        help="The gold dataset to use as seed",
        required=False,
    )

    parser.add_argument(
        "--training-file-name",
        type=str,
        default="generated_queries.jsonl",
        help="The training dataset",
        required=False,
    )

    return parser.parse_args()
