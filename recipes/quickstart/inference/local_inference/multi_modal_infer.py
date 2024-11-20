import argparse
import os
import sys
import torch
from accelerate import Accelerator
from PIL import Image as PIL_Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from peft import PeftModel
import gradio as gr
from huggingface_hub import HfFolder
# Initialize accelerator
accelerator = Accelerator()
device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MAX_OUTPUT_TOKENS = 2048
MAX_IMAGE_SIZE = (1120, 1120)


def get_hf_token():
    """Retrieve Hugging Face token from the cache or environment."""
    # Check if a token is explicitly set in the environment
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token

    # Automatically retrieve the token from the Hugging Face cache (set via huggingface-cli login)
    token = HfFolder.get_token()
    if token:
        return token

    print("Hugging Face token not found. Please login using `huggingface-cli login`.")
    sys.exit(1)


def load_model_and_processor(model_name: str, finetuning_path: str = None):
    """Load model and processor with optional LoRA adapter"""
    print(f"Loading model: {model_name}")
    hf_token = get_hf_token()
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map=device,
        token=hf_token
    )
    processor = MllamaProcessor.from_pretrained(model_name, token=hf_token, use_safetensors=True)

    if finetuning_path and os.path.exists(finetuning_path):
        print(f"Loading LoRA adapter from '{finetuning_path}'...")
        model = PeftModel.from_pretrained(
            model,
            finetuning_path,
            is_adapter=True,
            torch_dtype=torch.bfloat16
        )
        print("LoRA adapter merged successfully")
    
    model, processor = accelerator.prepare(model, processor)
    return model, processor

def process_image(image_path: str = None, image = None) -> PIL_Image.Image:
    """Process and validate image input"""
    if image is not None:
        return image.convert("RGB")
    if image_path and os.path.exists(image_path):
        return PIL_Image.open(image_path).convert("RGB")
    raise ValueError("No valid image provided")

def generate_text_from_image(model, processor, image, prompt_text: str, temperature: float, top_p: float):
    """Generate text from image using model"""
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(image, prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=MAX_OUTPUT_TOKENS)
    return processor.decode(output[0])[len(prompt):]

def gradio_interface(model_name: str):
    """Create Gradio UI with LoRA support"""
    # Initialize model state
    current_model = {"model": None, "processor": None}
    
    def load_or_reload_model(enable_lora: bool, lora_path: str = None):
        current_model["model"], current_model["processor"] = load_model_and_processor(
            model_name, 
            lora_path if enable_lora else None
        )
        return "Model loaded successfully" + (" with LoRA" if enable_lora else "")

    def describe_image(image, user_prompt, temperature, top_k, top_p, max_tokens, history):
        if image is not None:
            try:
                processed_image = process_image(image=image)
                result = generate_text_from_image(
                    current_model["model"],
                    current_model["processor"],
                    processed_image,
                    user_prompt,
                    temperature,
                    top_p
                )
                history.append((user_prompt, result))
            except Exception as e:
                history.append((user_prompt, f"Error: {str(e)}"))
        return history

    def clear_chat():
        return []

    with gr.Blocks() as demo:
        gr.HTML("<h1 style='text-align: center'>Llama Vision Model Interface</h1>")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model loading controls
                with gr.Group():
                    enable_lora = gr.Checkbox(label="Enable LoRA", value=False)
                    lora_path = gr.Textbox(
                        label="LoRA Weights Path",
                        placeholder="Path to LoRA weights folder",
                        visible=False
                    )
                    load_status = gr.Textbox(label="Load Status", interactive=False)
                    load_button = gr.Button("Load/Reload Model")

                # Image and parameter controls
                image_input = gr.Image(label="Image", type="pil", image_mode="RGB", height=512, width=512)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, value=0.6, step=0.1)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=100, value=50, step=1)
                top_p = gr.Slider(label="Top-p", minimum=0.1, maximum=1.0, value=0.9, step=0.1)
                max_tokens = gr.Slider(label="Max Tokens", minimum=50, maximum=MAX_OUTPUT_TOKENS, value=100, step=50)

            with gr.Column(scale=2):
                chat_history = gr.Chatbot(label="Chat", height=512)
                user_prompt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter your prompt",
                    lines=2
                )
                
                with gr.Row():
                    generate_button = gr.Button("Generate")
                    clear_button = gr.Button("Clear")

        # Event handlers
        enable_lora.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[enable_lora],
            outputs=[lora_path]
        )
        
        load_button.click(
            fn=load_or_reload_model,
            inputs=[enable_lora, lora_path],
            outputs=[load_status]
        )

        generate_button.click(
            fn=describe_image,
            inputs=[
                image_input, user_prompt, temperature,
                top_k, top_p, max_tokens, chat_history
            ],
            outputs=[chat_history]
        )
        
        clear_button.click(fn=clear_chat, outputs=[chat_history])

    # Initial model load
    load_or_reload_model(False)
    return demo

def main(args):
    """Main execution flow"""
    if args.gradio_ui:
        demo = gradio_interface(args.model_name)
        demo.launch()
    else:
        model, processor = load_model_and_processor(
            args.model_name,
            args.finetuning_path
        )
        image = process_image(image_path=args.image_path)
        result = generate_text_from_image(
            model, processor, image,
            args.prompt_text,
            args.temperature,
            args.top_p
        )
        print("Generated Text:", result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-modal inference with optional Gradio UI and LoRA support")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--prompt_text", type=str, help="Prompt text for the image")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--finetuning_path", type=str, help="Path to LoRA weights")
    parser.add_argument("--gradio_ui", action="store_true", help="Launch Gradio UI")
    
    args = parser.parse_args()
    main(args)
