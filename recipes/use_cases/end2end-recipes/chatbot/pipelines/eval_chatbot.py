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
from generator_utils import parse_qa_to_json, generate_LLM_eval

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
    score = bertscore.compute(
        predictions=generated,
        references=reference,
        lang="en"
    )
    f1 = score["f1"]
    precision = score["precision"]
    recall = score["recall"]
    return sum(precision)/len(precision), sum(recall)/len(recall), sum(f1)/len(f1)
# This function is used to eval the fine-tuned model, given the question, generate the answer.
async def eval_request(chat_service, api_context: dict, question: str) -> dict:
    prompt_for_system = api_context['eval_prompt_template'].format(language=api_context["language"])
    chat_request_payload = [{'role': 'system', 'content': prompt_for_system}, {'role': 'user', 'content': f"Question: {question}"}]
    # Getting a list of result, in this case, there should be only one result
    response_string = await chat_service.execute_chat_request_async(api_context, chat_request_payload)
    # convert the result string to a dict that contains Question, Answer
    result_list = parse_qa_to_json(response_string)
    if not result_list or len(result_list) > 1:
        print("Error: eval response should be a list of one result dict")
        return {}
    result = result_list[0]
    if "Answer" not in result:
        print("Error: eval response does not contain answer")
        return {}
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
        judge_list = []
        for index, item in enumerate(generated_answers):
            judge_list.append({"Question":questions[index],"Ground_truth":groud_truth[index],"Generated_answer":generated_answers[index]})
        if context["judge_endpoint"]:
            # make a copy of the context then change the VLLM endpoint to judge_endpoint
            context_copy = dict(context)
            context_copy["endpoint"] = context["judge_endpoint"]
            context_copy["model"] = "meta-llama/Meta-Llama-3-70B-Instruct"
            judge_results = await generate_LLM_eval(chat_service, context_copy, judge_list)
            correct_num = 0
            for result in judge_results:
                correct_num += result["Result"] == "YES"
            LLM_judge_score = correct_num/len(judge_results)
            print(f"The accuracy of the model is {LLM_judge_score}")
        rouge_score = compute_rouge_score(generated_answers,groud_truth)
        print("Rouge_score:",rouge_score)
        P, R, F1 = compute_bert_score(generated_answers,groud_truth)
        print(f"BERTScore Precision: {P:.4f}, Recall: {R:.4f}, F1: {F1:.4f}")
        # Saving the eval result to a log file
        with open(context["output_log"],"a") as fp:
            fp.write(f"Eval_result for {context['model']} \n")
            fp.write(f"Rouge_score: {rouge_score} \n")
            fp.write(f"BERTScore Precision: {P:.4f}, Recall: {R:.4f}, F1: {F1:.4f} \n")
            if context["judge_endpoint"]:
                fp.write(f"LLM_judge_score: {LLM_judge_score} \n")
            fp.write(f"QA details: \n")
            for item in judge_list:
                fp.write(f"question: {item['Question']} \n")
                fp.write(f"generated_answers: {item['Generated_answer']} \n")
                fp.write(f"groud_truth: {item['Ground_truth']} \n")
                fp.write("\n")
        logging.info(f"Eval successfully, the eval result is saved to {context['output_log']}.")
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
    parser.add_argument(
        "-j", "--judge_endpoint",
        default=None,
        type=int,
        help="If a port is specified, then use local vllm endpoint as judge LLM."
    )
    parser.add_argument(
        "-o", "--output_log",
        default="eval_result.log",
        help="save the eval result to a log file. Default is eval_result.log"
    )
    return parser.parse_args()

if __name__ == "__main__":
    logging.info("Initializing the process and loading configuration...")
    args = parse_arguments()
    context = load_config(args.config_path)
    context["model"] = args.model
    context["endpoint"] = args.vllm_endpoint
    context["judge_endpoint"] = args.judge_endpoint
    context["output_log"] = args.output_log
    if context["endpoint"]:
        logging.info(f"Use local vllm service for eval at port: '{args.vllm_endpoint}'.")
    if context["judge_endpoint"]:
        logging.info(f"Use local vllm service for judge at port: '{args.judge_endpoint}'.")
    asyncio.run(main(context))
