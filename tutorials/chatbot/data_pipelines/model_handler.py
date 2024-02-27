import os
import openai
import asyncio
from functools import partial
import json
from token_processor import split_string_by_token_length
from file_handler import get_file_string


# Throttling to manage rate limits
model_rate_limits = 100
max_concurrent_requests = int(model_rate_limits * 0.75)
throttler = asyncio.Semaphore(max_concurrent_requests)

async def send_chat_async(context: dict, request):
    async with throttler:
        try:
            loop = asyncio.get_running_loop()
            # Wrap the synchronous OpenAI API call with partial to pass arguments
            func = partial(
                openai.ChatCompletion.create,
                model=context['model'],
                messages=request,
                temperature=0.0
            )
            # Run the synchronous function in a separate thread
            resp = await loop.run_in_executor(None, func)
            # Process the response as before
            return next((msg['message']['content'] for msg in resp.choices if msg['message']['role'] == 'assistant'), "")
        except Exception as e:
            print(f"Error in send_chat_async: {e}")
            return ""

async def request_question(context: dict, input_str: str, num_data: int) -> dict:
    system_prompt = context['question_generator'].format(num_data=num_data, language=context["language"])
    request = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': input_str}]
    return json.loads(await send_chat_async(context, request))

async def generate_questions(context: dict):
    
    doc_string = get_file_string(context)
    batches = split_string_by_token_length(context, doc_string)
    num_questions_per_batch = context["num_data"] // len(batches)

    tasks = []
    for idx, batch in enumerate(batches):
        num_questions = num_questions_per_batch + 1 if idx == len(batches) - 1 and len(batches) * num_questions_per_batch < context["num_data"] else num_questions_per_batch
        tasks.append(request_question(context, batch, num_questions))

    results = await asyncio.gather(*tasks)
    return results

