import os
from PIL import Image as PIL_Image
import torch
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from accelerate import Accelerator
import gradio as gr

# Initialize accelerator
accelerator = Accelerator()

device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def load_model_and_processor(model_name: str, hf_token: str):
    """
    Load the model and processor based on the 11B or 90B model.
    """
    model = MllamaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, use_safetensors=True, 
                                                            device_map=device, token=hf_token)
    processor = MllamaProcessor.from_pretrained(model_name, token=hf_token, use_safetensors=True)

    model, processor = accelerator.prepare(model, processor)
    return model, processor


def process_image(image) -> PIL_Image.Image:
    """
    Open and convert an uploaded image to RGB format.
    """
    return image.convert("RGB")


def generate_text_from_image(model, processor, image, prompt_text: str, temperature: float, top_p: float):
    """
    Generate text from an image using the model and processor.
    """
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(image, prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=512)
    return processor.decode(output[0])[len(prompt):]


def inference(image, prompt_text, temperature, top_p):
    """
    Wrapper function to load the model and generate text based on inputs from Gradio UI.
    """
    hf_token = os.getenv("HF_TOKEN")  # Get the Hugging Face token from the environment
    if hf_token is None:
        return "Error: Hugging Face token not found in environment. Please set the HF_TOKEN environment variable."
    
    model, processor = load_model_and_processor(DEFAULT_MODEL, hf_token)
    processed_image = process_image(image)
    result = generate_text_from_image(model, processor, processed_image, prompt_text, temperature, top_p)
    return result


# Gradio UI
def create_gradio_interface():
    """
    Create the Gradio interface for image-to-text generation.
    """
    # Define the input components
    image_input = gr.Image(type="pil", label="Upload Image")
    prompt_input = gr.Textbox(lines=2, placeholder="Enter your prompt text", label="Prompt")
    temperature_input = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, label="Temperature")
    top_p_input = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, label="Top P")

    # Define the output component
    output_text = gr.Textbox(label="Generated Text")

    # Create the interface
    interface = gr.Interface(
        fn=inference,
        inputs=[image_input, prompt_input, temperature_input, top_p_input],
        outputs=output_text,
        title="LLama-3.2 Vision-Instruct",
        description="Generate descriptive text from an image using the LLama model.",
        theme="default",
    )
    
    # Launch the Gradio interface
    interface.launch()


if __name__ == "__main__":
    create_gradio_interface()
