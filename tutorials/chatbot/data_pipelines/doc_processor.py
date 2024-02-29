# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import tiktoken

# Assuming result_average_token is a constant, use UPPER_CASE for its name to follow Python conventions
AVERAGE_TOKENS_PER_RESULT = 100

def get_token_limit_for_model(model: str) -> int:
    """Returns the token limit for a given model."""
    if model == "gpt-3.5-turbo-16k":
        return 16384
    # Consider adding an else statement or default return value if more models are expected

def fetch_encoding_for_model(model="gpt-3.5-turbo-16k"):
    """Fetches the encoding for the specified model."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: Model not found. Using 'cl100k_base' encoding as default.")
        return tiktoken.get_encoding("cl100k_base")

def calculate_num_tokens_for_message(message: str, model="gpt-3.5-turbo-16k") -> int:
    """Calculates the number of tokens used by a message."""
    encoding = fetch_encoding_for_model(model)
    # Added 3 to account for priming with assistant's reply, as per original comment
    return len(encoding.encode(message)) + 3


def split_text_into_chunks(context: dict, text: str) -> list[str]:
    """Splits a long text into substrings based on token length constraints, adjusted for question generation."""
    # Adjusted approach to calculate max tokens available for text chunks
    model_token_limit = get_token_limit_for_model(context["model"])
    tokens_for_questions = calculate_num_tokens_for_message(context["question_prompt_template"])
    estimated_tokens_per_question = AVERAGE_TOKENS_PER_RESULT
    estimated_total_question_tokens = estimated_tokens_per_question * context["total_questions"]
    
    # Ensure there's a reasonable minimum chunk size
    max_tokens_for_text = max(model_token_limit - tokens_for_questions - estimated_total_question_tokens, model_token_limit // 10)
    
    encoded_text = fetch_encoding_for_model(context["model"]).encode(text)
    chunks, current_chunk = [], []
    print(f"Splitting text into chunks of {max_tokens_for_text} tokens, encoded_text {len(encoded_text)}", flush=True)
    for token in encoded_text:
        if len(current_chunk) + 1 > max_tokens_for_text:
            chunks.append(fetch_encoding_for_model(context["model"]).decode(current_chunk).strip())
            current_chunk = [token]
        else:
            current_chunk.append(token)
    
    if current_chunk:
        chunks.append(fetch_encoding_for_model(context["model"]).decode(current_chunk).strip())
    
    return chunks
