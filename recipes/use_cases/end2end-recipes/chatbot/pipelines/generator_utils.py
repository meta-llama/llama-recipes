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
def remove_non_printable(s):
    printable = set(string.printable)
    return ''.join(filter(lambda x: x in printable, s))
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
    with open(context['data_dir'] + '/' + 'all_text.txt', 'w') as f:
        f.write(text)
    return remove_non_printable(text)
# clean the text by removing all parts that did not contain any alphanumeric characters
def clean(s):
        result = []
        for item in s.split('"'):
            if any(c.isalnum() for c in item):
                result.append(item)
        return " ".join(result)
# given a response string, return a string that can be saved as json.
def parse_qac_to_json(response_string):
    split_lines = response_string.split("\n")
    start,mid,end = None,None,None
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
                mid = i
            elif '"Context":' in line:
                end = i
            # found Question means we have reached the end of the question, so add it to qa_list
            elif '"Question":' in line:
                question = " ".join(split_lines[start:mid]).split('"Question":')[1]
                answer = " ".join(split_lines[mid:end]).split('"Answer":')[1]
                context = " ".join(split_lines[end:i]).split('"Context":')[1]
                start,mid,end = i,None,None
                qa_set.add((clean(question), clean(answer),clean(context)))
        # adding last question back to qa_list
    if start and mid and end:
        question = " ".join(split_lines[start:mid]).split('"Question":')[1]
        answer = " ".join(split_lines[mid:end]).split('"Answer":')[1]
        context = " ".join(split_lines[end:]).split('"Context":')[1]
        start,mid,end = i,None,None
        qa_set.add((clean(question), clean(answer),clean(context)))
    qa_list = [{"Question": q, "Answer":a, "Context":c} for q,a,c in qa_set]

    return qa_list

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
    qa_list = [{"Question": q, "Answer":a} for q,a in qa_set]

    return qa_list

async def prepare_and_send_request(chat_service, api_context: dict, document_content: str, num_questions: int) -> dict:
    if num_questions == 0:
        logging.info(f"Error: num_questions is 0")
        return {}
    prompt_for_system = api_context['question_prompt_template'].format(num_questions=num_questions, language=api_context["language"])
    chat_request_payload = [{'role': 'system', 'content': prompt_for_system}, {'role': 'user', 'content': document_content}]
    # parse the result string to a list of dict that has Question, Answer, Context
    return await chat_service.execute_chat_request_async(api_context, chat_request_payload)
# This function is used to evaluate the quality of generated QA pairs. Return the original QA pair if the model eval result is YES. Otherwise, return an empty dict.
async def data_curation_request(chat_service, api_context: dict, document_content: dict) -> dict:
    prompt_for_system = api_context['curation_prompt_template'].format(language=api_context["language"])
    chat_request_payload = [{'role': 'system', 'content': prompt_for_system}, {'role': 'user', 'content': f"Question: {document_content['Question']} \n Answer: {document_content['Answer']}\n Context: {document_content['Context']} "}]
    result = await chat_service.execute_chat_request_async(api_context, chat_request_payload)
    if not result:
        return {}
    # no parsing needed, just return the loads the result as a dict
    result = json.loads(result)
    if "Result" not in result:
        print("Error: eval response does not contain answer")
        print(document_content,result)
        return {}
    # Send back the original QA pair is the model eval result is YES
    if result["Result"] == "YES":
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
            task = prepare_and_send_request(chat_service, api_context, batch_content, questions_in_current_batch)
            generation_tasks.append(task)
        except Exception as e:
            print(f"Error during chat request execution: {e}")

    question_generation_results = await asyncio.gather(*generation_tasks)
    final_result = []
    for result in question_generation_results:
        parsed_json = parse_qac_to_json(result)
        final_result.extend(parsed_json)
    return final_result

async def generate_data_curation(chat_service, api_context: dict, evaluation_list: list):
    eval_tasks = []
    for batch_index, batch_content in enumerate(evaluation_list):
        try:
            result = data_curation_request(chat_service, api_context, batch_content)
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
