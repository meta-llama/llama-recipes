import argparse
import os
import sys

import torch
from accelerate import Accelerator
from PIL import Image as PIL_Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor

accelerator = Accelerator()

device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def load_model_and_processor(model_name: str):
    """
    Load the model and processor based on the 11B or 90B model.
    """
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map=device,
    )
    processor = MllamaProcessor.from_pretrained(model_name, use_safetensors=True)

    model, processor = accelerator.prepare(model, processor)
    return model, processor


def process_image(image_path: str) -> PIL_Image.Image:
    """
    Open and convert an image from the specified path.
    """
    if not os.path.exists(image_path):
        print(f"The image file '{image_path}' does not exist.")
        sys.exit(1)
    with open(image_path, "rb") as f:
        return PIL_Image.open(f).convert("RGB")


def generate_text_from_image(
    model, processor, image, prompt_text: str, temperature: float, top_p: float
):
    """
    Generate text from an image using the model and processor.
    """
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }
    ]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(image, prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs, temperature=temperature, top_p=top_p, max_new_tokens=512
    )
    return processor.decode(output[0])[len(prompt) :]


def main(
    image_path: str, prompt_text: str, temperature: float, top_p: float, model_name: str
):
    """
    Call all the functions.
    """
    model, processor = load_model_and_processor(model_name)
    image = process_image(image_path)
    result = generate_text_from_image(
        model, processor, image, prompt_text, temperature, top_p
    )
    print("Generated Text: " + result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text from an image and prompt using the 3.2 MM Llama model."
    )
    parser.add_argument("--image_path", type=str, help="Path to the image file")
    parser.add_argument(
        "--prompt_text", type=str, help="Prompt text to describe the image"
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
        args.image_path, args.prompt_text, args.temperature, args.top_p, args.model_name
    )
