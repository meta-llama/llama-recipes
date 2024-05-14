# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import re
from transformers import  AutoTokenizer
import asyncio
import magic
from PyPDF2 import PdfReader
import json
from doc_processor import split_text_into_chunks
import logging
import json
# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    text = ' '.join(file_strings)
    if len(text) == 0:
        logging.error(f"Error reading files, text is empty")
    return ' '.join(file_strings)
# clean the text by removing all parts that did not contain any alphanumeric characters
def clean(s):
        result = []
        for item in s.split('"'):
            if any(c.isalnum() for c in item):
                result.append(item)
        return " ".join(result)

def parse_qa_to_json(response_string):
    split_lines = response_string.split("\n")
    start,end = None,None
    # must use set to avoid duplicate question/answer pairs due to async function calls
    qa_set = set()
    for i in range(len(split_lines)):
        line = split_lines[i]
        # starting to find "Question"
        if not start:
            # Once found, set start to this line number
            if '"Question":' in line:
                start = i
        else:
            # "Question" has been found, find "Answer", once found, set end to this line number
            if '"Answer":' in line:
                end = i
            # found Question means we have reached the end of the question, so add it to qa_list
            elif '"Question":' in line:
                question = " ".join(split_lines[start:end]).split('"Question":')[1]
                answer = " ".join(split_lines[end:i]).split('"Answer":')[1]
                start,end = i,None
                qa_set.add((clean(question), clean(answer)))
        # adding last question back to qa_list
    if start and end:
        question = " ".join(split_lines[start:end]).split('"Question":')[1]
        answer = " ".join(split_lines[end:]).split('"Answer":')[1]
        qa_set.add((clean(question), clean(answer)))
    qa_list = [{"question": q, "answer":a} for q,a in qa_set]
    return json.dumps(qa_list, indent=4)


async def prepare_and_send_request(chat_service, api_context: dict, document_content: str, num_questions: int) -> dict:
    prompt_for_system = api_context['question_prompt_template'].format(num_questions=num_questions, language=api_context["language"])
    chat_request_payload = [{'role': 'system', 'content': prompt_for_system}, {'role': 'user', 'content': document_content}]
    result = await chat_service.execute_chat_request_async(api_context, chat_request_payload,eval=False)
    if not result:
        return {}
    return json.loads(await chat_service.execute_chat_request_async(api_context, chat_request_payload,eval=False))
# This function is used to evaluate the quality of generated QA pairs. Return the original QA pair if the model eval result is YES. Otherwise, return an empty dict.
async def data_eval_request(chat_service, api_context: dict, document_content: dict) -> dict:
    prompt_for_system = api_context['eval_prompt_template'].format(language=api_context["language"])
    chat_request_payload = [{'role': 'system', 'content': prompt_for_system}, {'role': 'user', 'content': f"Question: {document_content['question']}, Answer: {document_content['answer']}"}]
    result = await chat_service.execute_chat_request_async(api_context, chat_request_payload,eval=True)
    if not result:
        return {}
    if "Answer" not in result:
        print("Error: eval response does not contain answer")
        print(document_content,result)
        return {}
    # Send back the original QA pair is the model eval result is YES
    if result["Answer"] == "YES":
        return document_content
    else:
        print(document_content,result)
    return {}


async def generate_question_batches(chat_service, api_context: dict):
    document_text = read_file_content(api_context)
    if len(document_text)== 0:
        logging.error(f"Error reading files, document_text is empty")
    if api_context["model"] in ["meta-llama-3-70b-instruct","meta-llama-3-8b-instruct"]:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", pad_token="</s>", padding_side="right")
    else:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", pad_token="</s>", padding_side="right")
    document_batches = split_text_into_chunks(api_context, document_text, tokenizer)

    total_questions = api_context["total_questions"]
    batches_count = len(document_batches)
    # each batch should have at least 1 question
    base_questions_per_batch = max(total_questions // batches_count,1)
    extra_questions = total_questions % batches_count

    print(f"Questions per batch: {base_questions_per_batch} (+1 for the first {extra_questions} batches), Total questions: {total_questions}, Batches: {batches_count}")
    generation_tasks = []
    for batch_index, batch_content in enumerate(document_batches):
        print(f"len of batch_content: {len(batch_content)}, batch_index: {batch_index}")
        #Distribute extra questions across the first few batches
        questions_in_current_batch = base_questions_per_batch + (1 if batch_index < extra_questions else 0)
        print(f"Batch {batch_index + 1} - {questions_in_current_batch} questions ********")
        try:
            result = prepare_and_send_request(chat_service, api_context, batch_content, questions_in_current_batch)
            generation_tasks.append(result)
        except Exception as e:
            print(f"Error during chat request execution: {e}")

    question_generation_results = await asyncio.gather(*generation_tasks)

    return question_generation_results

async def generate_data_eval(chat_service, api_context: dict, generated_questions: list):
    eval_tasks = []
    for batch_index, batch_content in enumerate(generated_questions):
        try:
            result = data_eval_request(chat_service, api_context, batch_content)
            eval_tasks.append(result)
        except Exception as e:
            print(f"Error during data eval request execution: {e}")

    eval_results = await asyncio.gather(*eval_tasks)
    curated_data = []
    for item in eval_results:
        # if the item is not empty, add it to the curated data list
        if item:
            curated_data.append(item)
    return curated_data
