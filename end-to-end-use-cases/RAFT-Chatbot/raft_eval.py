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
from langchain_community.vectorstores.utils import DistanceStrategy
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader
import re
import string
import pandas as pd 


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
        context += "\n<DOCUMENT>" + str(doc.page_content) + "</DOCUMENT>\n"
    return context
def build_retriever(api_config,embedding_model_name,retrieved_docs_num=5):
    # Use langchain to load the documents from data directory
    loader = DirectoryLoader(api_config['data_dir'])
    docs = loader.load()
    # Split the document into chunks with a specified chunk size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=api_config["chunk_size"],chunk_overlap=int(api_config["chunk_size"] / 10),separators= ["----------","\n\n", "\n", " ", ""],strip_whitespace=True)
    docs_processed = text_splitter.split_documents(docs)
    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)
    logging.info(f"Total number of docs_processed used by vectorstore: {len(docs_processed_unique)}")
    # Store the document into a vector store with a specific embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )
    vectorstore = FAISS.from_documents(docs_processed_unique, embedding_model, distance_strategy=DistanceStrategy.COSINE)
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": retrieved_docs_num},
    )
    return retriever
def generate_answers_with_RAG(model_name, question_list,api_config,retriever,api_url_overwrite=None):
    api_url = api_config['model_endpoint_url']
    if api_url_overwrite:
        api_url = api_url_overwrite
    key = api_config['api_key']
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
        # retrive the top K documents
        retrieved_docs = retriever.invoke(q)        
        # format the documents into a string
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
    judge_responses = ["YES" in item.content for item in judge_responses]
    correct_num = sum(judge_responses)
    return correct_num/len(questions),judge_responses
def score_single(api_config,generated,reference,questions, run_exact_match=True,run_rouge=True, run_llm_as_judge=True):
    # set metric to default -1, means no metric is computed
    metric = {
        "Rouge_score": -1,
        "LLM_judge_score": -1,
        "Exact_match": -1
    }
    if run_rouge:
        rouge_score = compute_rouge_score(generated,reference)
        metric["Rouge_score"] = rouge_score
        print("Rouge_score:",rouge_score)
    if api_config["judge_endpoint_url"] and run_llm_as_judge:
        api_url = api_config["judge_endpoint_url"]
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
        api_url = api_config["model_endpoint_url"]
        logging.info("Starting to generate answer given the eval set.")
        questions,groud_truth = [],[]
        if api_config["eval_file"].endswith(".parquet"):
            eval_file = pd.read_parquet(api_config["eval_file"],filters=[('source', '=', 'pt_discuss_forum')])
            for index, item in eval_file.iterrows():
                questions.append(item["question"]+"\nDetails:\n"+item["context"])
                groud_truth.append(item["answer"])
        else:
            with open(api_config["eval_file"]) as fp:
                eval_file = json.load(fp)
                for index, item in enumerate(eval_file):
                    questions.append(item["question"])
                    groud_truth.append(item["answer"])
        generated_answers = {}            
        # build retriver
        retriever = build_retriever(api_config,"sentence-transformers/multi-qa-mpnet-base-cos-v1",api_config["rag_topk"])
        # Generate answers for 8B models
        model_name = api_config["model_name"]
        generated_answers[model_name] = generate_answers_model_only(model_name,questions,api_url)
        generated_answers[model_name+"_RAG"] = generate_answers_with_RAG(model_name, questions,api_config,retriever)
        print("Finished generating answers for ", model_name)
        large_model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
        large_api_url = api_config["judge_endpoint_url"]
        generated_answers["70B_Base"] = generate_answers_model_only(large_model_name,questions,large_api_url)
        generated_answers["70B_RAG"] = generate_answers_with_RAG(large_model_name, questions,api_config,retriever,large_api_url)
        print("Finished generating answers for ", large_model_name)
        logging.info(f"Successfully generated {len(generated_answers[model_name+'_RAG'])} answers for all models.")
        # for generate answer from each model, compute the score metric
        all_metrics = []
        output_file = api_config["output_log"]+str(datetime.now().strftime("%Y%m%d_%H%M%S"))

        for model_name,model_answer in generated_answers.items():
            if len(model_answer) != len(groud_truth):
                print(f"The length of {model_name} answer is not equal to the length of ground truth.")
                continue
            metric = score_single(api_config,model_answer,groud_truth,questions)
            print(f"The eval result for {model_name} is: {metric}")
            with open(output_file,"a") as fp:
                fp.write(f"Eval_result for {model_name} \n")
                fp.write(f"Rouge_score: {metric['Rouge_score']} \n")
                fp.write(f"Exact_match_percentage: {metric['Exact_match']} \n")
                judge_responses = ["None"] * len(questions)
                if api_config["judge_endpoint_url"]:
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
        with open(output_file,"a") as fp:
            for item in all_metrics:
                fp.write(f"Model_Name: {item[0]}, LLM_SCORE: {item[1]} \n")
            for idx,item in enumerate(judge_zip):
                # if all the responses are "YES", then we skip this question
                if sum(item) == len(item):
                    continue 
                else:
                    fp.write(f"Comparing interested question: {questions[idx]} \n")
                    fp.write(f"groud_truth: {groud_truth[idx]} \n")
                    for i in range(len(model_names)):
                        fp.write(f"{item[i]} {model_names[i]}_answers: {generated_answers[model_names[i]][idx]} \n")
                    fp.write("------------------------\n")
            fp.write(json.dumps(all_metrics))
        print("Finished evaluating the model.")


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
        "-m", "--model_name",
        default=None,
        help="Provide the model_name to use for evaluation. If not specified, the model_path in eval_config.yaml will be used."
    )
    parser.add_argument(
        "-c", "--config_path",
        default="raft_eval_config.yaml",
        help="Set the configuration file path that has system prompt along with language, evalset path."
    )
    parser.add_argument(
        "-d", "--data_dir",
        default=None,
        help="Provide the data folder path to build RAG for evaluation. If not specified, the data_dir in eval_config.yaml will be used."
    )
    parser.add_argument(
        "-u", "--model_endpoint_url",
        default="http://localhost:8000/v1",
        type=str,
        help="The raft model endpoint url for eval."
    )
    parser.add_argument(
        "-j", "--judge_endpoint_url",
        default=None,
        type=str,
        help="The large model endpoint url for judge as LLM."
    )
    parser.add_argument(
        "-o", "--output_log",
        default="./eval_result",
        help="save the eval result to a log file. Default is eval_result[timestamp].log"
    )
    parser.add_argument(
        "-k", "--api_key",
        default="EMPTY",
        type=str,
        help="LLM API key for generating question/answer pairs."
    )
    parser.add_argument(
        "-r", "--rag_topk",
        default=5,
        type=int,
        help="set the number of top k documents the RAG needs to retrive."
    )
    parser.add_argument("--chunk_size", type=int, default=1000, help="The character size of each chunk used in RAG")
    return parser.parse_args()

if __name__ == "__main__":
    logging.info("Initializing the process and loading configuration...")
    args = parse_arguments()
    api_config = load_config(args.config_path)
    api_config["model_endpoint_url"] = args.model_endpoint_url
    if args.data_dir:
        api_config["data_dir"] = args.data_dir
    if args.model_name:
        api_config["model_name"] = args.model_name
    api_config["judge_endpoint_url"] = args.judge_endpoint_url
    api_config["output_log"] = args.output_log
    api_config["api_key"] = args.api_key
    api_config["chunk_size"] = args.chunk_size
    api_config["rag_topk"] = args.rag_topk
    if api_config["judge_endpoint_url"]:
        logging.info(f"The judge model url is: '{args.judge_endpoint_url}'.")
    main(api_config)
