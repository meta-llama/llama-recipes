import os
import sys
import argparse
from PIL import Image as PIL_Image
import torch
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from accelerate import Accelerator
from peft import PeftModel  # Make sure to install the `peft` library

accelerator = Accelerator()
device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def load_model_and_processor(model_name: str, hf_token: str, finetuning_path: str = None):
    """
    Load the model and processor, and optionally load adapter weights if specified.
    """
    # Load pre-trained model and processor
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        use_safetensors=True, 
        device_map=device,
        token=hf_token
    )
    processor = MllamaProcessor.from_pretrained(
        model_name, 
        token=hf_token, 
        use_safetensors=True
    )

    # If a finetuning path is provided, load the adapter model
    if finetuning_path and os.path.exists(finetuning_path):
        adapter_weights_path = os.path.join(finetuning_path, "adapter_model.safetensors")
        adapter_config_path = os.path.join(finetuning_path, "adapter_config.json")

        if os.path.exists(adapter_weights_path) and os.path.exists(adapter_config_path):
            print(f"Loading adapter from '{finetuning_path}'...")
            # Load the model with adapters using `peft`
            model = PeftModel.from_pretrained(
                model,
                finetuning_path,  # This should be the folder containing the adapter files
                is_adapter=True,
                torch_dtype=torch.bfloat16
            )

            print("Adapter merged successfully with the pre-trained model.")
        else:
            print(f"Adapter files not found in '{finetuning_path}'. Using pre-trained model only.")
    else:
        print(f"No fine-tuned weights or adapters found in '{finetuning_path}'. Using pre-trained model only.")

    # Prepare the model and processor for accelerated training
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


def generate_text_from_image(model, processor, image, prompt_text: str, temperature: float, top_p: float):
    """
    Generate text from an image using the model and processor.
    """
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(image, prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=2048)
    return processor.decode(output[0])[len(prompt):]


def main(image_path: str, prompt_text: str, temperature: float, top_p: float, model_name: str, hf_token: str, finetuning_path: str = None):
    """
    Call all the functions and optionally merge adapter weights from a specified path.
    """
    model, processor = load_model_and_processor(model_name, hf_token, finetuning_path)
    image = process_image(image_path)
    result = generate_text_from_image(model, processor, image, prompt_text, temperature, top_p)
    print("Generated Text: " + result)


if __name__ == "__main__":
    # Example usage with argparse (optional)
    parser = argparse.ArgumentParser(description="Generate text from an image using a fine-tuned model with adapters.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--prompt_text", type=str, required=True, help="Prompt text for the image.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="Pre-trained model name.")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API token.")
    parser.add_argument("--finetuning_path", type=str, help="Path to the fine-tuning weights (adapters).")
    
    args = parser.parse_args()

    main(
        image_path=args.image_path,
        prompt_text=args.prompt_text,
        temperature=args.temperature,
        top_p=args.top_p,
        model_name=args.model_name,
        hf_token=args.hf_token,
        finetuning_path=args.finetuning_path
    )
