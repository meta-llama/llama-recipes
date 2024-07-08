import jsonlines

from util.make_llama_3_prompt import make_llama_3_prompt


def load_training_data(args, make_question):
    path = f"data/training_data/{args.training_file_name}"

    limit = 1000

    with jsonlines.open(path) as reader:
        for index, obj in enumerate(reversed(list(reader))):
            if index >= limit:
                break

            yield {
                "input": make_llama_3_prompt(**make_question(obj)),
                "output": obj["sql"] + "<|eot_id|>",
            }


def get_dataset(args, make_question):
    dataset = list(load_training_data(args, make_question))
    return dataset
