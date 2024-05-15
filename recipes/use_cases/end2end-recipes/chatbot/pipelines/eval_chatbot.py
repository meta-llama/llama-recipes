# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.
from chat_utils import OctoAIChatService, VllmChatService
import logging
import evaluate
import argparse
from config import load_config
import asyncio
import json
from itertools import chain

def compute_rouge_score(generated : str, reference: str):
    rouge_score = evaluate.load('rouge')
    return rouge_score.compute(
        predictions=generated,
        references=reference,
        use_stemmer=True,
        use_aggregator=True
    )
def compute_bert_score(generated : str, reference: str):
    bertscore = evaluate.load("bertscore")
    return bertscore.compute(
        predictions=generated,
        references=reference,
        lang="en"
    )
# This function is used to evaluate the quality of generated QA pairs. Return the original QA pair if the model eval result is YES. Otherwise, return an empty dict.
async def eval_request(chat_service, api_context: dict, question: str) -> dict:
    prompt_for_system = api_context['eval_prompt_template'].format(language=api_context["language"])
    chat_request_payload = [{'role': 'system', 'content': prompt_for_system}, {'role': 'user', 'content': f"Question: {question}"}]
    # Getting a list of result, in this case, there should be only one result
    results = await chat_service.execute_chat_request_async(api_context, chat_request_payload,eval=False)
    # convert the result string to a list
    results = eval(results)
    if not results or len(results) > 1:
        print("results",type(results),len(results),results)
        return {}
    result = results[0]
    if "Answer" not in result:
        print("Error: eval response does not contain answer")
        print(question,result)
        return {}
    print("result",result)
    # Send back the model generated answer
    return result["Answer"]

async def generate_eval_answer(chat_service, api_context: dict, questions: list):
    eval_tasks = []
    for batch_index, question in enumerate(questions):
        try:
            result = eval_request(chat_service, api_context, question)
            eval_tasks.append(result)
        except Exception as e:
            print(f"Error during data eval request execution: {e}")
    print(len(eval_tasks),"eval_tasks")
    eval_results = await asyncio.gather(*eval_tasks)

    return eval_results

async def main(context):
    if context["endpoint"]:
        chat_service = VllmChatService()
    else:
        chat_service = OctoAIChatService()
    try:
        logging.info("Starting to generate answer given the eval set.")
        with open(context["eval_json"]) as fp:
            eval_json = json.load(fp)
        questions,groud_truth = [],[]
        for index, item in enumerate(eval_json):
            questions.append(item["question"])
            groud_truth.append(item["answer"])
        generated_answers = await generate_eval_answer(chat_service, context,questions)
        if not generated_answers:
            logging.warning("No answers generated. Please check the input context or model configuration.")
            return
        logging.info(f"Successfully generated {len(generated_answers)} answers.")
        rouge_score = compute_rouge_score(generated_answers,groud_truth)
        print("Rouge_score:",rouge_score)
        bert_score = compute_bert_score(generated_answers,groud_truth)
        print("Bert_score:",bert_score)
        logging.info("Eval successfully")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the process: {e}",exc_info=True)

def parse_arguments():
    # Define command line arguments for the script
    parser = argparse.ArgumentParser(
        description="Generate question/answer pairs from documentation."
    )
    parser.add_argument(
        "-m", "--model",
        default="chatbot",
        help="Select the model to use for evaluation, this maybe a LoRA adapter."
    )
    parser.add_argument(
        "-c", "--config_path",
        default="eval_config.yaml",
        help="Set the configuration file path that has system prompt along with language, evalset path."
    )
    parser.add_argument(
        "-v", "--vllm_endpoint",
        default=None,
        type=int,
        help="If a port is specified, then use local vllm endpoint for evaluations."
    )
    return parser.parse_args()

if __name__ == "__main__":
    logging.info("Initializing the process and loading configuration...")
    args = parse_arguments()

    context = load_config(args.config_path)
    context["model"] = args.model
    context["endpoint"] = args.vllm_endpoint
    if context["endpoint"]:
        logging.info(f"Use local vllm service at port: '{args.vllm_endpoint}'.")
    asyncio.run(main(context))
