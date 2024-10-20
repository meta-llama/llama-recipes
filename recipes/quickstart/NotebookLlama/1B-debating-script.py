import argparse

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

accelerator = Accelerator()
device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def load_model_and_tokenizer(model_name: str):
    """
    Load the model and tokenizer for LLaMA-1B.
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


def generate_response(model, tokenizer, conversation, temperature: float, top_p: float):
    """
    Generate a response based on the conversation history.
    """
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs, temperature=temperature, top_p=top_p, max_new_tokens=256
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt) :].strip()


def debate(
    model1,
    model2,
    tokenizer,
    system_prompt1,
    system_prompt2,
    initial_topic,
    n_turns,
    temperature,
    top_p,
):
    """
    Conduct a debate between two models.
    """
    conversation1 = [
        {"role": "system", "content": system_prompt1},
        {"role": "user", "content": f"Let's debate about: {initial_topic}"},
    ]
    conversation2 = [
        {"role": "system", "content": system_prompt2},
        {"role": "user", "content": f"Let's debate about: {initial_topic}"},
    ]

    for i in range(n_turns):
        print(f"\nTurn {i+1}:")

        # Model 1's turn
        response1 = generate_response(
            model1, tokenizer, conversation1, temperature, top_p
        )
        print(f"Model 1: {response1}")
        conversation1.append({"role": "assistant", "content": response1})
        conversation2.append({"role": "user", "content": response1})

        # Model 2's turn
        response2 = generate_response(
            model2, tokenizer, conversation2, temperature, top_p
        )
        print(f"Model 2: {response2}")
        conversation2.append({"role": "assistant", "content": response2})
        conversation1.append({"role": "user", "content": response2})


def main(
    system_prompt1: str,
    system_prompt2: str,
    initial_topic: str,
    n_turns: int,
    temperature: float,
    top_p: float,
    model_name: str,
):
    """
    Set up and run the debate.
    """
    model1, tokenizer = load_model_and_tokenizer(model_name)
    model2, _ = load_model_and_tokenizer(model_name)  # We can reuse the tokenizer

    debate(
        model1,
        model2,
        tokenizer,
        system_prompt1,
        system_prompt2,
        initial_topic,
        n_turns,
        temperature,
        top_p,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conduct a debate between two LLaMA-1B models."
    )
    parser.add_argument(
        "--system_prompt1",
        type=str,
        default="You are a passionate advocate for technology and innovation.",
        help="System prompt for the first model (default: 'You are a passionate advocate for technology and innovation.')",
    )
    parser.add_argument(
        "--system_prompt2",
        type=str,
        default="You are a cautious critic of rapid technological change.",
        help="System prompt for the second model (default: 'You are a cautious critic of rapid technological change.')",
    )
    parser.add_argument(
        "--initial_topic", type=str, required=True, help="Initial topic for the debate"
    )
    parser.add_argument(
        "--n_turns",
        type=int,
        default=5,
        help="Number of turns in the debate (default: 5)",
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
        args.system_prompt1,
        args.system_prompt2,
        args.initial_topic,
        args.n_turns,
        args.temperature,
        args.top_p,
        args.model_name,
    )
