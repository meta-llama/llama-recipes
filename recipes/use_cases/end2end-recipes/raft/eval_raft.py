# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.
import logging
import evaluate
import argparse
from config import load_config
import json
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import re
import string


def generate_answers_model_only(model_name,question_list,api_url="http://localhost:8000/v1",key="EMPTY"):
        # Use langchain to load the documents from data directory
    # Load the RAFT model

    llm = ChatOpenAI(
        openai_api_key=key,
        openai_api_base=api_url,
        model_name=model_name,
        temperature=0.0,
        max_tokens=1000
        )

    all_tasks = [api_config['eval_prompt_template'].format(question=question) for question in question_list]
    generated_answers = llm.batch(all_tasks)
    generated_answers = [ item.content for item in generated_answers]
    if len(generated_answers) == 0:
        logging.error("No model answers generated. Please check the input context or model configuration in ",model_name)
        return []
    return clean_text_list(generated_answers)
def format_docs_raft(docs):
    context = ""
    for doc in docs:
        context += "<DOCUMENT>" + str(doc.page_content) + "</DOCUMENT>\n"
    return context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def generate_answers_with_RAG(model_name, question_list,api_config,api_url_overwrite=None):
    data_dir = api_config['data_dir']
    api_url = "http://localhost:"+str(api_config['vllm_endpoint'])+"/v1"
    if api_url_overwrite:
        api_url = api_url_overwrite
    key = api_config['api_key']
    # Use langchain to load the documents from data directory
    loader = DirectoryLoader(data_dir)
    docs = loader.load()
    # Split the document into chunks with a specified chunk size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=api_config["chunk_size"], chunk_overlap=int(api_config["chunk_size"]/10))
    all_splits = text_splitter.split_documents(docs)

    # Store the document into a vector store with a specific embedding model
    vectorstore = FAISS.from_documents(all_splits, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",model_kwargs={'device': 'cuda'}))
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )
    # Load the RAFT model
    llm = ChatOpenAI(
        openai_api_key=key,
        openai_api_base=api_url,
        model_name=model_name,
        temperature=0.0,
        max_tokens=1000
        )
    all_tasks = []
    for q in question_list:
        # retrive the top 6 documents
        retrieved_docs = retriever.invoke(q)
        # format the documents into a string
        if '8B-Instruct' in model_name:
            documents = format_docs(retrieved_docs)
        else:
            documents = format_docs_raft(retrieved_docs)
        # create a prompt
        text = api_config["RAG_prompt_template"].format(context=documents,question=q)
        all_tasks.append(text)
    generated_answers = llm.batch(all_tasks)
    generated_answers = [ item.content for item in generated_answers]
    if len(generated_answers) == 0:
        logging.error("No RAG answers generated. Please check the input context or model configuration in ",model_name)
        return []
    return clean_text_list(generated_answers)
def compute_rouge_score(generated : list, reference: list):
    rouge_score = evaluate.load('rouge')
    return rouge_score.compute(
        predictions=generated,
        references=reference,
        use_stemmer=True,
        use_aggregator=True
    )
def clean_text_list(text_list):
    result = []
    for text in text_list:
        # for raft model, the answer will started with <ANSWER>
        index = text.rfind("<ANSWER>")
        if index!= -1:
            text = text[index:]
            text = text.replace("</ANSWER>:","")
        text = text.replace("begin_quote","")
        text = text.replace("end_quote","")
        text = text.replace("##","")
        text = text.strip()
        result.append(text)
    return result

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
def compute_judge_score(questions: list, generated : list, reference: list, api_config,api_url="http://localhost:8001/v1",key="EMPTY"):
    correct_num = 0
    model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
    llm = ChatOpenAI(
        openai_api_key=key,
        openai_api_base=api_url,
        model_name=model_name,
        max_tokens=1000,
        temperature=0.0)
    all_tasks = []
    for question,prediction,gold in zip(questions, generated,reference):
        message = api_config['judge_prompt_template'].format(question=question,prediction=prediction,gold=gold)
        all_tasks.append(message)
    judge_responses = llm.batch(all_tasks)
    judge_responses = ["YES" in item.content.split("<ANSWER>")[-1] for item in judge_responses]
    correct_num = sum(judge_responses)
    return correct_num/len(questions),judge_responses
def score_single(api_config,generated,reference,questions, run_exact_match=True,run_rouge=True, run_bert=True, run_llm_as_judge=True):
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
    if api_config["judge_endpoint"] and run_llm_as_judge:
        api_url = "http://localhost:"+str(api_config["judge_endpoint"])+"/v1"
        LLM_judge_score,judge_responses = compute_judge_score(questions, generated, reference, api_config,api_url=api_url)
        metric["LLM_judge_score"] = LLM_judge_score
        metric["LLM_judge_responses"] = judge_responses
        print(f"LLM_judge_score: {LLM_judge_score}")
    if run_exact_match:
        exact_match = exact_match_score(generated,reference)
        print(f"Exact_match_percentage: {exact_match:.4f}")
        metric["Exact_match"] = exact_match
    return metric
def main(api_config):
    # Since the eval set is small, we can run the eval without async functions
    try:
        api_url = "http://localhost:"+str(api_config["vllm_endpoint"])+"/v1"
        logging.info("Starting to generate answer given the eval set.")
        with open(api_config["eval_json"]) as fp:
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
            "70B_RAG": [],
            "70B_Base": [],
            
        }
        # Generate answers for baseline
        base_model_name = api_config["base_model_name"]
        generated_answers["Baseline"] = generate_answers_model_only(base_model_name,questions,api_url)
        generated_answers["Baseline_RAG"] = generate_answers_with_RAG(base_model_name, questions,api_config)
        # Generate answers for RAFT
        raft_model_name = api_config["raft_model_name"]
        generated_answers["RAFT"] = generate_answers_model_only(raft_model_name,questions,api_url)
        generated_answers["RAFT_RAG"] = generate_answers_with_RAG(raft_model_name, questions,api_config)

        large_model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
        large_api_url = "http://localhost:"+str(api_config["judge_endpoint"])+"/v1"
        generated_answers["70B_Base"] = generate_answers_model_only(large_model_name,questions,large_api_url)
        generated_answers["70B_RAG"] = generate_answers_with_RAG(large_model_name, questions,api_config,large_api_url,)
        logging.info(f"Successfully generated {len(generated_answers['Baseline_RAG'])} answers for all models.")
        # for generate answer from each model, compute the score metric
        all_metrics = []
        for model_name,model_answer in generated_answers.items():
            if len(model_answer) != len(groud_truth):
                print(f"The length of {model_name} answer is not equal to the length of ground truth.")
                continue
            metric = score_single(api_config,model_answer,groud_truth,questions)
            print(f"The eval result for {model_name} is: {metric}")
            with open(api_config["output_log"],"a") as fp:
                fp.write(f"Eval_result for {model_name} \n")
                fp.write(f"Rouge_score: {metric['Rouge_score']} \n")
                fp.write(f"BERTScore Precision: {metric['BERTScore_Precision']:.4f}, Recall: {metric['BERTScore_Recall']:.4f}, F1: {metric['BERTScore_F1']:.4f} \n")
                fp.write(f"Exact_match_percentage: {metric['Exact_match']} \n")
                judge_responses = ["None"] * len(questions)
                if api_config["judge_endpoint"]:
                    fp.write(f"LLM_judge_score: {metric['LLM_judge_score']} \n")
                    judge_responses = metric["LLM_judge_responses"]
                    all_metrics.append((model_name,metric['LLM_judge_score'],metric["LLM_judge_responses"]))
                fp.write(f"QA details: \n")
                for item in zip(questions,model_answer,groud_truth,judge_responses):
                    fp.write(f"question: {item[0]} \n")
                    fp.write(f"generated_answers: {item[1]} \n")
                    fp.write(f"groud_truth: {item[2]} \n")
                    fp.write(f"LLM_judge_response: {item[3]} \n")
                    fp.write("\n")
                fp.write("\n------------------------------------\n")
        # Now we want to take a closer look at the questions that are not answered the same by all the models.
        judge_zip = list(zip(*[item[-1] for item in all_metrics]))
        model_names = [item[0] for item in all_metrics]
        with open(api_config["output_log"],"a") as fp:
            for item in all_metrics:
                fp.write(f"Model_Name: {item[0]}, LLM_SCORE: {item[1]} \n")
            for idx,item in enumerate(judge_zip):
                # if all the responses are "YES" or all the responses are "NO", then we skip this question
                if sum([r=="YES" for r in item]) == len(item) or sum([r=="YES" for r in item]) == 0:
                    continue 
                else:
                    fp.write(f"Comparing interested question: {questions[idx]} \n")
                    fp.write(f"groud_truth: {groud_truth[idx]} \n")
                    for i in range(len(model_names)):
                        fp.write(f"{item[i]} {model_names[i]}_answers: {generated_answers[model_names[i]][idx]} \n")
                    fp.write("-------\n")



        logging.info(f"Eval successfully, the eval result is saved to {api_config['output_log']}.")
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
    parser.add_argument(
        "-k", "--api_key",
        default="EMPTY",
        type=str,
        help="LLM API key for generating question/answer pairs."
    )
    parser.add_argument("--chunk_size", type=int, default=1000, help="The character size of each chunk used in RAG")
    return parser.parse_args()

if __name__ == "__main__":
    logging.info("Initializing the process and loading configuration...")
    args = parse_arguments()
    api_config = load_config(args.config_path)
    api_config["vllm_endpoint"] = args.vllm_endpoint
    if args.data_dir:
        api_config["data_dir"] = args.data_dir
    if args.raft_model_name:
        api_config["raft_model_name"] = args.raft_model_name
    api_config["judge_endpoint"] = args.judge_endpoint
    api_config["output_log"] = args.output_log
    api_config["api_key"] = args.api_key
    api_config["chunk_size"] = args.chunk_size
    if api_config["judge_endpoint"]:
        logging.info(f"Use local vllm service for judge at port: '{args.judge_endpoint}'.")
    main(api_config)
