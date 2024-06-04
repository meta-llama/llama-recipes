# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.
from chat_utils import OctoAIChatService, VllmChatService
import logging
import evaluate
import argparse
from config import load_config
import json
from itertools import chain
from langchain_community.llms import VLLMOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, SystemMessage
import re
import string
from collections import Counter

def generate_answers_model_only(model_name,question_list,api_url="http://localhost:8000/v1",key="EMPTY"):
        # Use langchain to load the documents from data directory
    # Load the RAFT model
    llm = VLLMOpenAI(
        openai_api_key=key,
        openai_api_base=api_url,
        model_name=model_name,
        model_kwargs={"stop": ["."]},
        temperature=0.0,)
    generated_answers = []
    for question in question_list:
        response = llm.invoke(question)
        generated_answers.append(response)
    if len(generated_answers) == 0:
        logging.error("No model answers generated. Please check the input context or model configuration in ",model_name)
        return []
    return generated_answers
def generate_answers_with_RAG(model_name, data_dir,question_list, api_url="http://localhost:8000/v1",key="EMPTY"):
    # Use langchain to load the documents from data directory
    loader = DirectoryLoader(data_dir)
    docs = loader.load()
    # Split the document into chunks with a specified chunk size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_splits = text_splitter.split_documents(docs)

    # Store the document into a vector store with a specific embedding model
    vectorstore = FAISS.from_documents(all_splits, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
    # Load the RAFT model
    llm = VLLMOpenAI(
        openai_api_key=key,
        openai_api_base=api_url,
        model_name=model_name,
        model_kwargs={"stop": ["."]},
        temperature=0.0,)
    # Create a RetrievalQA chain with the vector store and RAFT model
    qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever()
    )
    generated_answers = []
    for question in question_list:
        response = qa_chain({"query": question})
        generated_answers.append(response['result'])
    if len(generated_answers) == 0:
        logging.error("No RAG answers generated. Please check the input context or model configuration in ",model_name)
        return []
    return generated_answers
def compute_rouge_score(generated : list, reference: list):
    rouge_score = evaluate.load('rouge')
    return rouge_score.compute(
        predictions=generated,
        references=reference,
        use_stemmer=True,
        use_aggregator=True
    )
def remove_special_tokens(text_list):
    clean_text_list = []
    for text in text_list:
        text = text.replace("##begin_quote##","")
        text = text.replace("##end_quote##","")
        text = text.strip()
        clean_text_list.append(text)
    return clean_text_list

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
def exact_match_score(prediction, ground_truth):
    """Computes EM score for a single prediction and ground truth answer."""
    num_match = 0
    assert len(prediction) == len(ground_truth), "Answer length does not match prediction length."
    assert(len(ground_truth) > 0)
    for idx, (pred,gold) in enumerate(zip(prediction, ground_truth)):
        if (normalize_answer(pred) == normalize_answer(gold)):
            num_match += 1
    return num_match/len(ground_truth)
def compute_bert_score(generated : list, reference: list):
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
def compute_judge_score(questions: list, generated : list, reference: list, context,api_url="http://localhost:8001/v1",key="EMPTY"):
    correct_num = 0
    model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
    llm = VLLMOpenAI(
        openai_api_key=key,
        openai_api_base=api_url,
        model_name=model_name,
        model_kwargs={"stop": ["."]},
        temperature=0.0,)
    for q,pred,gold in zip(questions, generated,reference):
        # messages = [
        #     SystemMessage(content=context['judge_prompt_template']),
        #     HumanMessage(content=f"Question: {q} \n Teacher's Answer: {gold} \n Student's Answer: {pred} "),
        # ]
        messages = context['judge_prompt_template'] + "\n"
        messages += f"Question: {q} \n Teacher's Answer: {gold} \n Student's Answer: {pred} "
        response = llm.invoke(messages)
        print(response+ " -------------")
        result = json.loads(response)
        if "Result" not in result:
            print("Error: eval response does not contain answer")
            print(result)
            continue
        correct_num += result["Result"] == "YES"
    return correct_num/len(questions)
def score_single(context,generated,reference,questions, run_exact_match=True,run_rouge=True, run_bert=True, run_llm_as_judge=True):
    # set metric to default -1, means no metric is computed
    metric = {
        "Rouge_score": -1,
        "BERTScore_Precision": -1,
        "BERTScore_Recall": -1,
        "BERTScore_F1": -1,
        "LLM_judge_score": -1,
        "Exact_match": -1
    }
    if run_rouge:
        rouge_score = compute_rouge_score(generated,reference)
        metric["Rouge_score"] = rouge_score
        print("Rouge_score:",rouge_score)
    if run_bert:
        P, R, F1 = compute_bert_score(generated,reference)
        print(f"BERTScore Precision: {P:.4f}, Recall: {R:.4f}, F1: {F1:.4f}")
        metric["BERTScore_Precision"] = P
        metric["BERTScore_Recall"] = R
        metric["BERTScore_F1"] = F1
    if context["judge_endpoint"] and run_llm_as_judge:
        api_url = "http://localhost:"+str(context["judge_endpoint"])+"/v1"
        LLM_judge_score = compute_judge_score(questions, generated, reference, context,api_url=api_url)
        metric["LLM_judge_score"] = LLM_judge_score
        print(f"LLM_judge_score: {LLM_judge_score}")
    if run_exact_match:
        exact_match = exact_match_score(generated,reference)
        print(f"Exact_match_percentage: {exact_match:.4f}")
        metric["Exact_match"] = exact_match
    return metric
def main(context):
    # Since the eval set is small, we can run the eval without async functions
    try:
        api_url = "http://localhost:"+str(context["vllm_endpoint"])+"/v1"
        logging.info("Starting to generate answer given the eval set.")
        with open(context["eval_json"]) as fp:
            eval_json = json.load(fp)
        questions,groud_truth = [],[]
        for index, item in enumerate(eval_json):
            questions.append(item["question"])
            groud_truth.append(item["answer"])
        generated_answers = {
            "RAFT": [],
            "RAFT_RAG": [],
            "Baseline": [],
            "Baseline_RAG": [],
        }
        # Generate answers for baseline
        base_model_name = context["base_model_name"]
        generated_answers["Baseline"] = generate_answers_model_only(base_model_name,questions,api_url)
        #generated_answers["Baseline_RAG"] = generate_answers_with_RAG(base_model_name, context["data_dir"],questions,api_url)
        # Generate answers for RAFT
        raft_model_name = context["raft_model_name"]
        #generated_answers["RAFT"] = generate_answers_model_only(raft_model_name,questions,api_url)
        #generated_answers["RAFT_RAG"] = generate_answers_with_RAG(raft_model_name, context["data_dir"],questions,api_url)
        # clean special tokens from the RAFT generated answer
        #generated_answers["RAFT"] = remove_special_tokens(generated_answers["RAFT"])
        #generated_answers["RAFT_RAG"] = remove_special_tokens(generated_answers["RAFT_RAG"])
        logging.info(f"Successfully generated {len(generated_answers['Baseline_RAG'])} answers for all models.")
        # for generate answer from each model, compute the score metric
        for model_name,model_answer in generated_answers.items():
            if len(model_answer) != len(groud_truth):
                print(f"The length of {model_name} answer is not equal to the length of ground truth.")
                continue
            metric = score_single(context,model_answer,groud_truth,questions)
            print(f"The eval result for {model_name} is: {metric}")
            with open(context["output_log"],"a") as fp:
                fp.write(f"Eval_result for {model_name} \n")
                fp.write(f"Rouge_score: {metric['Rouge_score']} \n")
                fp.write(f"BERTScore Precision: {metric['BERTScore_Precision']:.4f}, Recall: {metric['BERTScore_Recall']:.4f}, F1: {metric['BERTScore_F1']:.4f} \n")
                fp.write(f"Exact_match_percentage: {metric['Exact_match']} \n")
                if context["judge_endpoint"]:
                    fp.write(f"LLM_judge_score: {metric['LLM_judge_score']} \n")
                fp.write(f"QA details: \n")
                for item in zip(questions,model_answer,groud_truth):
                    fp.write(f"question: {item[0]} \n")
                    fp.write(f"generated_answers: {item[1]} \n")
                    fp.write(f"groud_truth: {item[2]} \n")
                    fp.write("\n")
                fp.write("\n------------------------------------\n")
        logging.info(f"Eval successfully, the eval result is saved to {context['output_log']}.")
        # Saving the eval result to a log file
    except Exception as e:
        logging.error(f"An unexpected error occurred during the process: {e}",exc_info=True)

def parse_arguments():
    # Define command line arguments for the script
    parser = argparse.ArgumentParser(
        description="Generate question/answer pairs from documentation."
    )
    parser.add_argument(
        "-m", "--raft_model_name",
        default=None,
        help="Provide the raft_model_name to use for evaluation. If not specified, the model_path in eval_config.yaml will be used."
    )
    parser.add_argument(
        "-c", "--config_path",
        default="eval_config.yaml",
        help="Set the configuration file path that has system prompt along with language, evalset path."
    )
    parser.add_argument(
        "-d", "--data_dir",
        default=None,
        help="Provide the data folder path to build RAG for evaluation. If not specified, the data_dir in eval_config.yaml will be used."
    )
    parser.add_argument(
        "-v", "--vllm_endpoint",
        default=8000,
        type=int,
        help="If a port is specified, then use local vllm endpoint for eval."
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
    context["vllm_endpoint"] = args.vllm_endpoint
    if args.data_dir:
        context["data_dir"] = args.data_dir
    if args.raft_model_name:
        context["raft_model_name"] = args.raft_model_name
    context["judge_endpoint"] = args.judge_endpoint
    context["output_log"] = args.output_log
    if context["judge_endpoint"]:
        logging.info(f"Use local vllm service for judge at port: '{args.judge_endpoint}'.")
    main(context)
