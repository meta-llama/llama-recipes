DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

import argparse

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

accelerator = Accelerator()
device = accelerator.device


def load_model_and_tokenizer(model_name: str):
    """
    Load the model and tokenizer for LLaMA-8b.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)

    model, tokenizer = accelerator.prepare(model, tokenizer)
    return model, tokenizer


def generate_text(model, tokenizer, conversation, temperature: float, top_p: float):
    """
    Generate text using the model and tokenizer based on a conversation.
    """
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs, temperature=temperature, top_p=top_p, max_new_tokens=512
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt) :]


def main(
    system_message: str,
    user_message: str,
    temperature: float,
    top_p: float,
    model_name: str,
):
    """
    Call all the functions.
    """
    model, tokenizer = load_model_and_tokenizer(model_name)
    conversation = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    result = generate_text(model, tokenizer, conversation, temperature, top_p)
    print("Generated Text: " + result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text using the LLaMA-8b model with system and user messages."
    )
    parser.add_argument(
        "--system_message",
        type=str,
        default="You are a helpful AI assistant.",
        help="System message to set the context (default: 'You are a helpful AI assistant.')",
    )
    parser.add_argument(
        "--user_message", type=str, required=True, help="User message for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top p for generation (default: 0.9)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name (default: '{DEFAULT_MODEL}')",
    )

    args = parser.parse_args()
    main(
        args.system_message,
        args.user_message,
        args.temperature,
        args.top_p,
        args.model_name,
    )
