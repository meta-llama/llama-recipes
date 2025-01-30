import string

import datasets


# def doc_to_text(doc: dict) -> str:
#     question, choice = doc["input_question"], str(doc["input_choice_list"])
#     prompt = f"<|start_header_id|>user<|end_header_id|>You are a helpful assistant designed to help with multiple choice question. Think step by step. Question: {question}\n {choice}\n<|eot_id|> \n\n<|start_header_id|>assistant<|end_header_id|>"
#     return prompt


def doc_to_text(doc: dict) -> str:
    input_question, input_choice = doc["input_question"], str(doc["input_choice_list"])

    optimized_examples = [
        {
            "question": "Explain what difficulties would arise if messenger RNA molecules were not destroyed after they had produced some polypeptide chains.",
            "options": {
                "A": "mRNA would replicate rapidly",
                "B": "The cell would use mRNA as a source of energy",
                "C": "The cell would lack proteins",
                "D": "Cell would enter a state of permanent division",
                "E": "mRNA would be transformed into DNA",
                "F": "Excess protein production, energy depletion, and potential harm to the cell",
                "G": "mRNA would exit the cell and infect neighboring cells",
                "H": "Proteins would be broken down into mRNA",
                "I": "mRNA would become part of the cell membrane",
                "J": "mRNA would bind to lipids and carbohydrates, disrupting cellular metabolism",
            },
            "answer": "F",
        },
        {
            "question": "Based on the characteristic population curves that result from plotting population growth of a species, the most effective means of controlling the mosquito population is to",
            "options": {
                "A": "opt for zero population control once the K value of the curve has been reached",
                "B": "maintain the population at the highest point of its logistic curve",
                "C": "reduce the carrying capacity cif the environment to lower the K value",
                "D": "decrease the mortality rate",
                "E": "increase the birth rate of the species",
                "F": "drastically reduce the population below the K value",
                "G": "maintain the population at a point corresponding to the midpoint of its logistic curve",
                "H": "increase the carrying capacity of the environment to raise the K value",
                "I": "opt for zero population control at the beginning of the logistic curve",
                "J": "null",
            },
            "answer": "C",
        },
        {
            "question": "Solve the equation 1.2 = 0.4y using mental math.",
            "options": {
                "A": "3",
                "B": "\u22123",
                "C": "0.8",
                "D": "2",
                "E": "0.3",
                "F": "5",
                "G": "\u22124",
                "H": "4",
                "I": "6",
                "J": "1.6",
            },
            "answer": "A",
        },
        {
            "question": "assume you are Indonesian. In 2010, the rupiah exchange rate was around IDR15,000/USD, and the consumer price index in Indonesia and the United States was at 100. In 2019, the exchange rate changed to IDR14,000/USD. Simultaneously, Indonesia\u2019s inflation rose 5% due to the consumer price index rising to 105. Meanwhile, the United States\u2019 inflation rate rose 10% due to the consumer price index rising to 110. Whats the real exchange rate?",
            "options": {
                "A": "14000.00",
                "B": "15500.00",
                "C": "15000.00",
                "D": "16000.00",
                "E": "13500.00",
                "F": "14666.67",
                "G": "13888.89",
                "H": "14800.00",
                "I": "15200.00",
                "J": "13333.33",
            },
            "answer": "F",
        },
        {
            "question": "To move the economy closer to full employment the central bank decides that the federal funds rate must be increased. The appropriate open market operation is to ______ which ______ the money supply ______ aggregate demand and fights ______. OMO \u00a0\u00a0\u00a0 MONEY SUPPLY \u00a0\u00a0\u00a0 AD \u00a0\u00a0\u00a0 TO FIGHT",
            "options": {
                "A": "Buy bonds \u00a0\u00a0\u00a0 Decreases \u00a0\u00a0\u00a0 Decreases \u00a0\u00a0\u00a0 Unemployment",
                "B": "Sell bonds \u00a0\u00a0\u00a0 Increases \u00a0\u00a0\u00a0 Decreases \u00a0\u00a0\u00a0 Unemployment",
                "C": "Buy bonds \u00a0\u00a0\u00a0 Increases \u00a0\u00a0\u00a0 Decreases \u00a0\u00a0\u00a0 Unemployment",
                "D": "Sell bonds \u00a0\u00a0\u00a0 Increases \u00a0\u00a0\u00a0 Decreases \u00a0\u00a0\u00a0 Inflation",
                "E": "Buy bonds \u00a0\u00a0\u00a0 Increases \u00a0\u00a0\u00a0 Increases \u00a0\u00a0\u00a0 Unemployment",
                "F": "Sell bonds \u00a0\u00a0\u00a0 Decreases \u00a0\u00a0\u00a0 Decreases \u00a0\u00a0\u00a0 Unemployment",
                "G": "Buy bonds \u00a0\u00a0\u00a0 Increases \u00a0\u00a0\u00a0 Increases \u00a0\u00a0\u00a0 Inflation",
                "H": "Sell bonds \u00a0\u00a0\u00a0 Decreases \u00a0\u00a0\u00a0 Decreases \u00a0\u00a0\u00a0 Inflation",
                "I": "Buy bonds \u00a0\u00a0\u00a0 Decreases \u00a0\u00a0\u00a0 Increases \u00a0\u00a0\u00a0 Inflation",
                "J": "Sell bonds \u00a0\u00a0\u00a0 Increases \u00a0\u00a0\u00a0 Increases \u00a0\u00a0\u00a0 Inflation",
            },
            "answer": "H",
        },
    ]
    formatted_text = ""
    for question in optimized_examples:
        # Format user message with question and options
        user_msg = "<|start_header_id|>user<|end_header_id|>\n\n"
        user_msg += "Given the following question and candidate answers, choose the best answer.\n"
        user_msg += f"Question: {question['question']}\n"

        # Add options
        for option_key, option_value in question["options"].items():
            if option_value is not None:
                user_msg += f"{option_key}. {option_value}\n"

        user_msg += '\nYour response should end with "The best answer is [the_answer_letter]." where the [the_answer_letter] is a letter from the provided choices.\n\n'
        user_msg += f"Let's think step by step. Question: {input_question}\n {input_choice}\n <|eot_id|>"

        formatted_text += user_msg

        # Add assistant placeholder message
    assistant_msg = "<|start_header_id|>assistant<|end_header_id|>\n\n<|eot_id|>"

    formatted_text += assistant_msg

    # print(formatted_text)
    # exit()

    return formatted_text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["input_question"],
            "gold": doc["input_correct_responses"][0],
        }
        return out_doc

    dataset = dataset.select(range(1200, len(dataset)))

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
