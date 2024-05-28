# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import re
import string
from transformers import  AutoTokenizer
import asyncio
import magic
from PyPDF2 import PdfReader
import json
from doc_processor import split_text_into_chunks
import logging
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from math import ceil
import random
# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def strip_str(s: str) -> str:
    """
    Helper function for helping format strings returned by GPT-4.
    """
    l, r = 0, len(s)-1
    beg_found = False
    for i in range(len(s)):
        if s[i].isalpha():
            if not beg_found:
                l = i
                beg_found = True
            else:
                r = i
    r += 2
    return s[l:min(r, len(s))]
def read_text_file(file_path):
    try:
        with open(file_path, 'r') as f:
            text = f.read().strip() + ' '
            if len(text) == 0:
                print("File is empty ",file_path)
            return text
    except Exception as e:
        logging.error(f"Error reading text file {file_path}: {e}")
    return ''

def read_pdf_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            num_pages = len(pdf_reader.pages)
            file_text = [pdf_reader.pages[page_num].extract_text().strip() + ' ' for page_num in range(num_pages)]
            text = ''.join(file_text)
            if len(text) == 0:
                print("File is empty ",file_path)
            return ''.join(file_text)
    except Exception as e:
        logging.error(f"Error reading PDF file {file_path}: {e}")
    return ''

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Assuming each item in the list has a 'question' and 'answer' key
            # Concatenating question and answer pairs with a space in between and accumulating them into a single string
            file_text = ' '.join([item['question'].strip() + ' ' + item['answer'].strip() + ' ' for item in data])
            if len(file_text) == 0:
                print("File is empty ",file_path)
            return file_text
    except Exception as e:
        logging.error(f"Error reading JSON file {file_path}: {e}")
    return ''


def process_file(file_path):
    print("starting to process file: ", file_path)
    file_type = magic.from_file(file_path, mime=True)
    if file_type in ['text/plain', 'text/markdown', 'JSON']:
        return read_text_file(file_path)
    elif file_type == 'application/pdf':
        return read_pdf_file(file_path)
    else:
        logging.warning(f"Unsupported file type {file_type} for file {file_path}")
        return ''
def read_file_content(context):
    file_strings = []

    for root, _, files in os.walk(context['data_dir']):
        for file in files:
            file_path = os.path.join(root, file)
            file_text = process_file(file_path)
            if file_text:
                file_strings.append(file_text)
    text = '\n'.join(file_strings)
    text = remove_non_printable(text)
    return remove_non_printable(text)

def remove_non_printable(s):
    printable = set(string.printable)
    return ''.join(filter(lambda x: x in printable, s))


async def generate_question_request(chat_service, api_context: dict, document_content: str, num_questions: int) -> dict:
    if num_questions == 0:
        logging.info(f"Error: num_questions is 0")
        return {}
    prompt_for_system = api_context['question_prompt_template'].format(num_questions=num_questions)
    chat_request_payload = [{'role': 'system', 'content': prompt_for_system}, {'role': 'user', 'content': str(document_content)}]
    # parse the result string to a list of dict that has Question, Answer, Context
    return await chat_service.execute_chat_request_async(api_context, chat_request_payload)

def get_chunks(
    text: str,
    chunk_size: int = 512,
    embedding_model: str = None
) -> list[str]:
    """
    Takes in a `file_path` and `doctype`, retrieves the document, breaks it down into chunks of size
    `chunk_size`, and returns the chunks.
    """
    chunks = []
    if  len(text) == 0:
        raise TypeError("Can not get chunks from empty text")
    else:
        num_chunks = ceil(len(text) / chunk_size)
        logging.info(f"Splitting text into {num_chunks} chunks")
        text_splitter = SemanticChunker(embedding_model, number_of_chunks=num_chunks)
        chunks = text_splitter.create_documents([text])
        chunks = [chunk.page_content for chunk in chunks]

    return chunks
# read all the files in the data folder, then split them into chunks
# generate questions for each chunk and return a list of questions list
async def generate_questions(chat_service, api_context: dict):
    document_text = read_file_content(api_context)
    if len(document_text)== 0:
        logging.error(f"Error reading files, document_text is empty")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    document_batches = get_chunks(document_text,api_context["chunk_size"],embedding_model)

    batches_count = len(document_batches)
    total_questions = api_context["questions_per_chunk"] * batches_count

    print(f"Questions per batch: {api_context['questions_per_chunk']}, Total questions: {total_questions}, Batches: {batches_count}")
    generation_tasks = []
    for batch_index, batch_content in enumerate(document_batches):
        print(f"len of batch_content: {len(batch_content)}, batch_index: {batch_index}")
        #Distribute extra questions across the first few batches
        print(f"Batch {batch_index + 1} - {api_context['questions_per_chunk']} questions ********")
        try:
            task = generate_question_request(chat_service, api_context, batch_content, api_context["questions_per_chunk"])
            generation_tasks.append(task)
        except Exception as e:
            print(f"Error during chat request execution: {e}")

    question_generation_results = await asyncio.gather(*generation_tasks)
    final_result = []
    for result in question_generation_results:
        queries = result.split('\n')
        queries = [strip_str(q) for q in queries]
        queries = [q for q in queries if any(c.isalpha() for c in q)]
        if len(queries) > int(api_context['questions_per_chunk']):
            # As the model may have unrelated question at the begining of the result
            # if queries is more than questions_per_chunk, then we need to truncate it and only keep last questions_per_chunk lines
            queries = queries[-int(api_context['questions_per_chunk']):]
        final_result.append(queries)
    return final_result

def add_chunk_to_dataset(
    chunks: list[str],
    chunk: str,
    x: int = 5,
    num_distract: int = 3,
    p: float = 0.8,
    model: str = None
) -> None:
    """
    Given a chunk, create {Q, A, D} triplets and add them to the dataset.
    """
    global ds
    i = chunks.index(chunk)
    qs = generate_instructions(client, chunk, x, model) if doctype == "api" else generate_instructions_gen(client, chunk, x, model)
    for q in qs:
        datapt = {
            "id": None,
            "type": None,
            "question": None,
            "context": None,
            "oracle_context": None,
            "cot_answer": None
        }

        datapt["id"] = f"seed_task_{0 if not ds else ds.num_rows}"
        datapt["type"] = "api call" if doctype == "api" else "general"
        datapt["question"] = q

        # add num_distract distractor docs
        docs = [chunk]
        indices = list(range(0, len(chunks)))
        indices.remove(i)
        for j in random.sample(indices, num_distract):
            docs.append(chunks[j])
        # decides whether to add oracle document
        oracle = random.uniform(0, 1) < p
        if not oracle:
            docs[0] = chunks[random.sample(indices, 1)[0]]
        random.shuffle(docs)

        d = {
            "title": [],
            "sentences": []
        }

        d["title"].append(["placeholder_title"]*(num_distract+1))
        d["sentences"].append(docs)
        datapt["context"] = d
        datapt["oracle_context"] = chunk

        # add answer to q
        datapt["cot_answer"] = generate_label(client, q, chunk, doctype, model=model)

        # construct model instruction
        context = ""
        for doc in docs:
            context += "<DOCUMENT>" + str(doc) + "</DOCUMENT>\n"
        context += q
        datapt["instruction"] = context

        # add to dataset
        if not ds:
            # init ds
            datapt["id"] = [datapt["id"]]
            datapt["type"] = [datapt["type"]]
            datapt["question"] = [datapt["question"]]
            datapt["context"] = [datapt["context"]]
            datapt["oracle_context"] = [datapt["oracle_context"]]
            datapt["cot_answer"] = [datapt["cot_answer"]]
            datapt["instruction"] = [datapt["instruction"]]
            ds = Dataset.from_dict(datapt)
        else:
            ds = ds.add_item(datapt)

# This function is used to evaluate the quality of generated QA pairs. Return the original QA pair if the model eval result is YES. Otherwise, return an empty dict.
async def LLM_judge_request(chat_service, api_context: dict, document_content: dict) -> dict:
    prompt_for_system = api_context['judge_prompt_template'].format(language=api_context["language"])
    chat_request_payload = [{'role': 'system', 'content': prompt_for_system}, {'role': 'user', 'content': f"Question: {document_content['Question']} \n Teacher's Answer: {document_content['Ground_truth']}\n Student's Answer: {document_content['Generated_answer']} "}]
    result = await chat_service.execute_chat_request_async(api_context, chat_request_payload)
    if not result:
        return {}
    # no parsing needed, just return the loads the result as a dict
    result = json.loads(result)
    if "Result" not in result:
        print("Error: eval response does not contain answer")
        print(document_content,result)
        return {}
    return result

async def generate_LLM_eval(chat_service, api_context: dict, judge_list: list):
    eval_tasks = []
    for batch_index, batch_content in enumerate(judge_list):
        try:
            result = LLM_judge_request(chat_service, api_context, batch_content)
            eval_tasks.append(result)
        except Exception as e:
            print(f"Error during data eval request execution: {e}")

    judge_results = await asyncio.gather(*eval_tasks)
    return judge_results
