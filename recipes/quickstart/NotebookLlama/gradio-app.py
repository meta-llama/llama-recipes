import gradio as gr
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

accelerator = Accelerator()
device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def load_model_and_tokenizer(model_name: str):
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
    conversation1 = [
        {"role": "system", "content": system_prompt1},
        {"role": "user", "content": f"Let's debate about: {initial_topic}"},
    ]
    conversation2 = [
        {"role": "system", "content": system_prompt2},
        {"role": "user", "content": f"Let's debate about: {initial_topic}"},
    ]

    debate_history = []

    for i in range(n_turns):
        # Model 1's turn
        response1 = generate_response(
            model1, tokenizer, conversation1, temperature, top_p
        )
        debate_history.append(f"Model 1: {response1}")
        conversation1.append({"role": "assistant", "content": response1})
        conversation2.append({"role": "user", "content": response1})
        yield "\n".join(debate_history)

        # Model 2's turn
        response2 = generate_response(
            model2, tokenizer, conversation2, temperature, top_p
        )
        debate_history.append(f"Model 2: {response2}")
        conversation2.append({"role": "assistant", "content": response2})
        conversation1.append({"role": "user", "content": response2})
        yield "\n".join(debate_history)


def create_gradio_interface():
    model1, tokenizer = load_model_and_tokenizer(DEFAULT_MODEL)
    model2, _ = load_model_and_tokenizer(DEFAULT_MODEL)  # We can reuse the tokenizer

    def gradio_debate(
        system_prompt1, system_prompt2, initial_topic, n_turns, temperature, top_p
    ):
        debate_generator = debate(
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
        debate_text = ""
        for turn in debate_generator:
            debate_text = turn
            yield debate_text

    iface = gr.Interface(
        fn=gradio_debate,
        inputs=[
            gr.Textbox(
                label="System Prompt 1",
                value="You are a passionate advocate for technology and innovation.",
            ),
            gr.Textbox(
                label="System Prompt 2",
                value="You are a cautious critic of rapid technological change.",
            ),
            gr.Textbox(
                label="Initial Topic",
                value="The impact of artificial intelligence on society",
            ),
            gr.Slider(minimum=1, maximum=10, step=1, label="Number of Turns", value=5),
            gr.Slider(
                minimum=0.1, maximum=1.0, step=0.1, label="Temperature", value=0.7
            ),
            gr.Slider(minimum=0.1, maximum=1.0, step=0.1, label="Top P", value=0.9),
        ],
        outputs=gr.Textbox(label="Debate", lines=20),
        title="LLaMA 1B Model Debate",
        description="Watch two LLaMA 1B models debate on a topic of your choice!",
        live=False,  # Changed to False to prevent auto-updates
    )
    return iface


if __name__ == "__main__":
    iface = create_gradio_interface()
    iface.launch(share=True)
