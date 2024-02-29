# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import openai
import asyncio
import magic
from PyPDF2 import PdfReader
from functools import partial
import json
from doc_processor import split_text_into_chunks
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Manage rate limits with throttling
rate_limit_threshold = 2000
allowed_concurrent_requests = int(rate_limit_threshold * 0.75)
request_limiter = asyncio.Semaphore(allowed_concurrent_requests)

def read_text_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return f.read().strip() + ' '
    except Exception as e:
        logging.error(f"Error reading text file {file_path}: {e}")
    return ''

def read_pdf_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            num_pages = len(pdf_reader.pages)
            file_text = [pdf_reader.pages[page_num].extract_text().strip() + ' ' for page_num in range(num_pages)]
            return ''.join(file_text)
    except Exception as e:
        logging.error(f"Error reading PDF file {file_path}: {e}")
    return ''

def process_file(file_path):
    file_type = magic.from_file(file_path, mime=True)
    if file_type in ['text/plain', 'text/markdown']:
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

    return ' '.join(file_strings)


async def execute_chat_request_async(api_context: dict, chat_request):
    async with request_limiter:
        try:
            event_loop = asyncio.get_running_loop()
            # Prepare the OpenAI API call
            openai_chat_call = partial(
                openai.ChatCompletion.create,
                model=api_context['model'],
                messages=chat_request,
                temperature=0.0
            )
            # Execute the API call in a separate thread
            response = await event_loop.run_in_executor(None, openai_chat_call)
            # Extract and return the assistant's response
            return next((message['message']['content'] for message in response.choices if message['message']['role'] == 'assistant'), "")
        except Exception as error:
            print(f"Error during chat request execution: {error}")
            return ""

async def prepare_and_send_request(api_context: dict, document_content: str, total_questions: int) -> dict:
    prompt_for_system = api_context['question_prompt_template'].format(total_questions=total_questions, language=api_context["language"])
    chat_request_payload = [{'role': 'system', 'content': prompt_for_system}, {'role': 'user', 'content': document_content}]
    return json.loads(await execute_chat_request_async(api_context, chat_request_payload))

async def generate_question_batches(api_context: dict):
    document_text = read_file_content(api_context)
    document_batches = split_text_into_chunks(api_context, document_text)
    
    total_questions = api_context["total_questions"]
    batches_count = len(document_batches)
    base_questions_per_batch = total_questions // batches_count
    extra_questions = total_questions % batches_count

    print(f"Questions per batch: {base_questions_per_batch} (+1 for the first {extra_questions} batches), Total questions: {total_questions}, Batches: {batches_count}")
    
    generation_tasks = []
    for batch_index, batch_content in enumerate(document_batches):
        # Distribute extra questions across the first few batches
        questions_in_current_batch = base_questions_per_batch + (1 if batch_index < extra_questions else 0)
        print(f"Batch {batch_index + 1} - {questions_in_current_batch} questions ********")
        generation_tasks.append(prepare_and_send_request(api_context, batch_content, questions_in_current_batch))

    question_generation_results = await asyncio.gather(*generation_tasks)

    return question_generation_results



