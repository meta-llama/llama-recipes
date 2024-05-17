import asyncio
import logging
from abc import ABC, abstractmethod
from octoai.client import OctoAI
from functools import partial
from openai import OpenAI
import json
# Configure logging to include the timestamp, log level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Since OctoAI has different naming for llama models, create this mapping to get huggingface offical model name given OctoAI names.
MODEL_NAME_MAPPING={"meta-llama-3-70b-instruct":"meta-llama/Meta-Llama-3-70B-Instruct",
"meta-llama-3-8b-instruct":"meta-llama/Meta-Llama-3-8B-Instruct","llama-2-7b-chat":"meta-llama/Llama-2-7b-chat-hf"
,"llama-2-70b-chat":"meta-llama/Llama-2-70b-chat-hf"}
# Manage rate limits with throttling
rate_limit_threshold = 2000
allowed_concurrent_requests = int(rate_limit_threshold * 0.75)
request_limiter = asyncio.Semaphore(allowed_concurrent_requests)
class ChatService(ABC):
    @abstractmethod
    async def execute_chat_request_async(self, api_context: dict, chat_request):
        pass

# Please implement your own chat service class here.
# The class should inherit from the ChatService class and implement the execute_chat_request_async method.
# The following are two example chat service classes that you can use as a reference.
class OctoAIChatService(ChatService):
    async def execute_chat_request_async(self, api_context: dict, chat_request):
        async with request_limiter:
            try:
                event_loop = asyncio.get_running_loop()
                client = OctoAI(api_context['api_key'])
                api_chat_call = partial(
                    client.chat.completions.create,
                    model=api_context['model'],
                    messages=chat_request,
                    temperature=0.0
                )
                response = await event_loop.run_in_executor(None, api_chat_call)
                assistant_response = next((choice.message.content for choice in response.choices if choice.message.role == 'assistant'), "")
                return assistant_response
            except Exception as error:
                logging.error(f"Error during chat request execution: {error}",exc_info=True)
                return ""
# Use the local vllm openai compatible server for generating question/answer pairs to make API call syntax consistent
# please read for more detail:https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html.
class VllmChatService(ChatService):
    async def execute_chat_request_async(self, api_context: dict, chat_request):
        try:
            event_loop = asyncio.get_running_loop()
            if api_context["model"] in MODEL_NAME_MAPPING:
                model_name = MODEL_NAME_MAPPING[api_context['model']]
            else:
                model_name = api_context['model']
            client = OpenAI(api_key=api_context['api_key'], base_url="http://localhost:"+ str(api_context['endpoint'])+"/v1")
            api_chat_call = partial(
                client.chat.completions.create,
                model=model_name,
                messages=chat_request,
                temperature=0.0
            )
            response = await event_loop.run_in_executor(None, api_chat_call)
            assistant_response = next((choice.message.content for choice in response.choices if choice.message.role == 'assistant'), "")
            return assistant_response
        except Exception as error:
            logging.error(f"Error during chat request execution: {error}",exc_info=True)
            return ""
