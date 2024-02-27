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

def split_text_into_tokenized_chunks(context: dict, text_to_split: str) -> list[str]:
    """Splits a long string into substrings based on token length constraints."""
    max_tokens_per_chunk = (
        get_token_limit_for_model(context["model"]) -
        calculate_num_tokens_for_message(context["question_prompt_template"]) -
        AVERAGE_TOKENS_PER_RESULT * context["total_questions"]
    )
    substrings = []
    chunk_tokens = []

    encoding = fetch_encoding_for_model(context["model"])
    text_tokens = encoding.encode(text_to_split)

    for token in text_tokens:
        if len(chunk_tokens) + 1 > max_tokens_per_chunk:
            substrings.append(encoding.decode(chunk_tokens).strip())
            chunk_tokens = [token]
        else:
            chunk_tokens.append(token)

    if chunk_tokens:
        substrings.append(encoding.decode(chunk_tokens).strip())

    return substrings
