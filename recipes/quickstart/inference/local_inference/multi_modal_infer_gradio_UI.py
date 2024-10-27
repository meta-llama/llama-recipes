import gradio as gr
import torch
import os
from PIL import Image
from accelerate import Accelerator
from transformers import MllamaForConditionalGeneration, AutoProcessor
import argparse  # Import argparse

# Parse the command line arguments
parser = argparse.ArgumentParser(description="Run Gradio app with Hugging Face model")
parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face authentication token")
args = parser.parse_args()

# Hugging Face token
hf_token = args.hf_token

# Initialize Accelerator
accelerate = Accelerator()
device = accelerate.device

# Set memory management for PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # or adjust size as needed

# Model ID
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Load model with the Hugging Face token
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device,
    use_auth_token=hf_token  # Pass the Hugging Face token here
)

# Load the processor
processor = AutoProcessor.from_pretrained(model_id, use_auth_token=hf_token)

# Visual theme
visual_theme = gr.themes.Default()  # Default, Soft or Monochrome

# Constants
MAX_OUTPUT_TOKENS = 2048
MAX_IMAGE_SIZE = (1120, 1120)

# Function to process the image and generate a description
def describe_image(image, user_prompt, temperature, top_k, top_p, max_tokens, history):
    # Initialize cleaned_output variable
    cleaned_output = ""

    if image is not None:
        # Resize image if necessary
        image = image.resize(MAX_IMAGE_SIZE)
        prompt = f"<|image|><|begin_of_text|>{user_prompt} Answer:"
        # Preprocess the image and prompt
        inputs = processor(image, prompt, return_tensors="pt").to(device)
    else:
        # Text-only input if no image is provided
        prompt = f"<|begin_of_text|>{user_prompt} Answer:"
        # Preprocess the prompt only (no image)
        inputs = processor(prompt, return_tensors="pt").to(device)

    # Generate output with model
    output = model.generate(
        **inputs,
        max_new_tokens=min(max_tokens, MAX_OUTPUT_TOKENS),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

    # Decode the raw output
    raw_output = processor.decode(output[0])

    # Clean up the output to remove system tokens
    cleaned_output = raw_output.replace("<|image|><|begin_of_text|>", "").strip().replace(" Answer:", "")

    # Ensure the prompt is not repeated in the output
    if cleaned_output.startswith(user_prompt):
        cleaned_output = cleaned_output[len(user_prompt):].strip()

    # Append the new conversation to the history
    history.append((user_prompt, cleaned_output))

    return history


# Function to clear the chat history
def clear_chat():
    return []

# Gradio Interface
def gradio_interface():
    with gr.Blocks(visual_theme) as demo:
        gr.HTML(
        """
    <h1 style='text-align: center'>
    meta-llama/Llama-3.2-11B-Vision-Instruct
    </h1>
    """)
        with gr.Row():
            # Left column with image and parameter inputs
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Image", 
                    type="pil", 
                    image_mode="RGB", 
                    height=512,  # Set the height
                    width=512   # Set the width
                )

                # Parameter sliders
                temperature = gr.Slider(
                    label="Temperature", minimum=0.1, maximum=1.0, value=0.6, step=0.1, interactive=True)
                top_k = gr.Slider(
                    label="Top-k", minimum=1, maximum=100, value=50, step=1, interactive=True)
                top_p = gr.Slider(
                    label="Top-p", minimum=0.1, maximum=1.0, value=0.9, step=0.1, interactive=True)
                max_tokens = gr.Slider(
                    label="Max Tokens", minimum=50, maximum=MAX_OUTPUT_TOKENS, value=100, step=50, interactive=True)

            # Right column with the chat interface
            with gr.Column(scale=2):
                chat_history = gr.Chatbot(label="Chat", height=512)

                # User input box for prompt
                user_prompt = gr.Textbox(
                    show_label=False,
                    container=False,
                    placeholder="Enter your prompt", 
                    lines=2
                )

                # Generate and Clear buttons
                with gr.Row():
                    generate_button = gr.Button("Generate")
                    clear_button = gr.Button("Clear")

                # Define the action for the generate button
                generate_button.click(
                    fn=describe_image, 
                    inputs=[image_input, user_prompt, temperature, top_k, top_p, max_tokens, chat_history],
                    outputs=[chat_history]
                )

                # Define the action for the clear button
                clear_button.click(
                    fn=clear_chat,
                    inputs=[],
                    outputs=[chat_history]
                )

    return demo

# Launch the interface
demo = gradio_interface()
# demo.launch(server_name="0.0.0.0", server_port=12003)
demo.launch()