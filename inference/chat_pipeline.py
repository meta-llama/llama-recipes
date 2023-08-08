# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import fire
import torch
import os
import sys
import warnings
from typing import List
from peft import PeftModel, PeftConfig
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from safety_utils import get_safety_checker
from model_utils import load_model, load_peft_model
# from chat_utils import read_dialogs_from_file, format_tokens, format_chat_tokens
from transformers import pipeline, Conversation, ConversationalPipeline
def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    min_new_tokens:int=0, #The minimum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility 
    safety_score_threshold: float=0.5,
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_saleforce_content_safety: bool=True, # Enable safety check woth Saleforce safety flan t5
    **kwargs
):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    model = LlamaForCausalLM.from_pretrained(model_name)
    tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast = False) # need to update the fast tokenizer file
    chatbot = ConversationalPipeline(model=model,tokenizer = tokenizer, device = 0, max_length=435,temperature=0.6,top_p=0.9)
    conversation = Conversation("What is so great about #1?", past_user_inputs = ["I am going to Paris, what should I see?"], generated_responses=["Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world."])
    conversation = chatbot(conversation)
    print(conversation.generated_responses[-1])
if __name__ == "__main__":
    fire.Fire(main)