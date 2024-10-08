import os
from PIL import Image as PIL_Image
import torch
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from accelerate import Accelerator
import gradio as gr
import gc  # Import garbage collector


# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"

class LlamaInference:
    def __init__(self, model_name=DEFAULT_MODEL, hf_token=None):
        """
        Initialize the inference class. Load model and processor.
        """
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        if self.hf_token is None:
            raise ValueError("Error: Hugging Face token not found in environment. Please set the HF_TOKEN environment variable.")

        # Load model and processor
        self.model, self.processor = self.load_model_and_processor()

    def load_model_and_processor(self):
        """
        Load the model and processor based on the model name.
        """
        model = MllamaForConditionalGeneration.from_pretrained(self.model_name, 
                                                               torch_dtype=torch.bfloat16, 
                                                               use_safetensors=True, 
                                                               device_map=self.device, 
                                                               token=self.hf_token)
        processor = MllamaProcessor.from_pretrained(self.model_name, 
                                                    token=self.hf_token, 
                                                    use_safetensors=True)

        # Prepare model and processor with accelerator
        model, processor = self.accelerator.prepare(model, processor)
        return model, processor

    def process_image(self, image) -> PIL_Image.Image:
        """
        Open and convert an uploaded image to RGB format.
        """
        return image.convert("RGB")

    def generate_text_from_image(self, image, prompt_text: str, temperature: float, top_p: float):
        """
        Generate text from an image using the model and processor.
        """
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)

        # Perform inference without computing gradients to save memory
        with torch.no_grad():
            output = self.model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=512)
        
        return self.processor.decode(output[0])[len(prompt):]

    def cleanup(self):
        """
        Clean up instance variables to release memory.
        """
        # Move model and processor to CPU before deleting to free up GPU memory
        self.model.to('cpu')
        del self.model
        del self.processor
        torch.cuda.empty_cache()  # Release GPU memory
        gc.collect()  # Force garbage collection

        # Clear other instance variables
        del self.accelerator
        del self.device
        del self.hf_token

        print("Cleanup complete. Instance variables deleted and memory cleared.")


def inference(image, prompt_text, temperature, top_p):
    """
    Main inference function to handle Gradio inputs and manage memory cleanup.
    """
    # Initialize the inference instance (this loads the model)
    llama_inference = LlamaInference()

    try:
        # Process the image and generate text
        processed_image = llama_inference.process_image(image)
        result = llama_inference.generate_text_from_image(processed_image, prompt_text, temperature, top_p)
    finally:
        # Perform memory cleanup
        llama_inference.cleanup()

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
