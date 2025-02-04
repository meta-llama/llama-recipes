import string

import datasets


def doc_to_text(doc: dict) -> str:
    question, choice = doc["input_question"], str(doc["input_choice_list"])
    prompt = [
        "You are a helpful assistant."
        "To answer these questions, carefully analyze the given scenarios or information, considering the context and applying relevant knowledge from various subjects such as law, morality, science, history, and critical thinking. Evaluate the options based on ordinary moral standards, factual information, or the characteristics of biological relationships. Choose the answer that best aligns with the analysis, ensuring that the reasoning is sound and the conclusion is supported by the provided information or general knowledge."
    ]
    default_parsing_text = "Regardless of the approach, always conclude with:\nThe best answer is [the_answer_letter].\nwhere the [the_answer_letter] is one of A, B, C or D."
    template = f"<|start_header_id|>user<|end_header_id|>{prompt[1]}. {default_parsing_text} Question: {question}\n {choice}\n<|eot_id|> \n\n<|start_header_id|>assistant<|end_header_id|>"
    return template


# def doc_to_text(doc: dict) -> str:
#     # Strip out the last two characters, which is a space and the answer
#     # E.g., "Answer: B" -> "Answer:"
#     return doc["input_final_prompts"][0][:-2]


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["input_question"],
            "gold": doc["input_correct_responses"][0],
        }
        return out_doc

    # dataset = dataset.select(range(1500, len(dataset)))

    dataset = dataset.select_columns(
        [
            "input_question",
            "input_correct_responses",
            "input_final_prompts",
            "is_correct",
            "input_question_hash",
            "input_choice_list",
            "output_prediction_text",
        ]
    )
    dataset = dataset.rename_column("is_correct", "previously_is_correct")
    dataset = dataset.map(_process_doc)
    return dataset.map(_process_doc)


def doc_to_target(doc: dict) -> str:
    return doc["gold"]
