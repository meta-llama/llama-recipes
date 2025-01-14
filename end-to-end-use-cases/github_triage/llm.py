import logging 
from typing import Any, Dict, List, Optional, Union
import yaml
import time
import json

from tqdm import tqdm
from openai import OpenAI
import groq

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
CFG = yaml.safe_load(open("config.yaml", "r"))

class LlamaVLLM():
    def __init__(self, endpoint, model_id):
        self.model_id = model_id
        self.client = OpenAI(base_url=endpoint, api_key='token')

    def chat(
        self,
        inputs: List[Dict[str, str]],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        guided_decode_json_schema: Optional[str] = None
    ) -> List[str]:

        if generation_kwargs is None:
            generation_kwargs = {}
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=inputs,
                extra_body={
                    "guided_json": guided_decode_json_schema
                },
                **generation_kwargs,
            )
            output = response.choices[0].message
        except Exception as e:
            logger.error(
                f"FAILED to generate inference for input {inputs}\nError: {str(e)}"
            )
            output = None
        return output
    

class LlamaGroq():
    def __init__(self, key, model_id):
        self.model_id = model_id
        self.client = groq.Groq(api_key=key)
        logger.debug(f"Using Groq:{self.model_id} for inference")

    def chat(
        self, 
        inputs: List[Dict[str, str]], 
        generation_kwargs: Optional[Dict[str, Any]] = None,
        guided_decode_json_schema: Optional[str] = None
    ) -> str:
        
        if generation_kwargs is None:
            generation_kwargs = {}
            
        # Currently Groq doesn't support guided JSON decoding. Workaround:
        if guided_decode_json_schema is not None:
            inputs[0]['content'] += f"\n\nEnsure your response aligns with the following JSON schema:\n{guided_decode_json_schema}\n\n"
        
        output = None
        
        while True:
            try:
                response = self.client.chat.completions.with_raw_response.create(
                    model=self.model_id,
                    messages=inputs,
                    stream=False,
                    **generation_kwargs,
                    response_format={"type": 'json_object' if guided_decode_json_schema is not None else 'text'}
                )
                completion = response.parse()
                output = completion.choices[0].message.content
                break
            except groq.RateLimitError as e:
                wait = e.response.headers['X-Ratelimit-Reset']
                response = e.response
                print(e)
                print(f"[groq] waiting for {wait} to prevent ratelimiting")
                time.sleep(wait)
            except Exception as e:
                logger.error(f"INFERENCE FAILED with Error: {e.response.status_code} for input:\n{inputs[-1]['content'][:300]}")
                break

        return output


def run_llm_inference(
    prompt_name: str,
    inputs: Union[str, List[str]],
    generation_kwargs: Optional[Dict] = None,
    guided_decode_json_schema=None,
) -> Union[List[str], List[Dict[str, Any]]]:
    """
    Run the LLM inference on the given inputs.

    Args:
    - prompt_name (str): The name of the prompt to use.
    - inputs (str or List[str]): The input(s) to the LLM.
    - generation_kwargs (Dict): Additional keyword arguments to pass to the LLM.
    - guided_decode_json_schema (str): The JSON schema to use for guided decoding.

    Returns:
    - Union[str, List[str]]: The response(s) from the LLM.
    """
    
    # initialize appropriate LLM accessor
    if CFG['model']['use'] == 'vllm':
        LLM = LlamaVLLM(**CFG['model']['vllm'])
    elif CFG['model']['use'] == 'groq':
        LLM = LlamaGroq(**CFG['model']['groq'])
    else:
        raise ValueError("Invalid model type in config.yaml")
    
    logger.debug(f"Running `{prompt_name}` inference with {CFG['model']['use']}")
    
    _batch = True
    if isinstance(inputs, str):
        _batch = False
        inputs = [inputs]

    inputs = [
        [
            {"role": "system", "content": CFG["prompts"][prompt_name]["system"]},
            {"role": "user", "content": i},
        ]
        for i in inputs
    ]

    if (
        guided_decode_json_schema is None
        and "json_schema" in CFG["prompts"][prompt_name]
    ):
        guided_decode_json_schema = " ".join(
            CFG["prompts"][prompt_name]["json_schema"].split()
        )

    responses = [
        LLM.chat(i, generation_kwargs, guided_decode_json_schema) 
        for i in tqdm(inputs, desc=f"Inference[{prompt_name}]")
    ]

    if guided_decode_json_schema is not None:
        responses_json = []
        for r in responses:
            if r is not None:
                try:
                    responses_json.append(json.loads(r, strict=False))
                    continue
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON: {r}")
            responses_json.append(None)
        responses = responses_json

    if not _batch:
        responses = responses[0]

    return responses
