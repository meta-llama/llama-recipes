import gradio as gr

from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]

def prompt_template_dropdown_listener(value):
    # This function will be called whenever the value of the dropdown changes
    if value == "Single Turn":
        return {
            assistant_response: gr.Textbox(elem_id="assistant_response", visible = False),
            user_prompt_2: gr.Textbox(elem_id="user_prompt_2", visible = False)
        }
    else:
        return {
            assistant_response: gr.Textbox(elem_id="assistant_response", visible = True),
            user_prompt_2: gr.Textbox(elem_id="user_prompt_2", visible = True)
        }

def format_prompt_template_listener(system_prompt, user_prompt_1, assistant_response, user_prompt_2):
    if not user_prompt_1:
        return {
            ""
        }
    dialog: Dialog = []
    if system_prompt: dialog.append({   "role": "system", "content": system_prompt, })
    dialog.append({   "role": "user", "content": user_prompt_1, })
    if assistant_response: dialog.append({   "role": "assistant", "content": assistant_response, })
    if user_prompt_2: dialog.append({   "role": "user", "content": user_prompt_2, })

    return ChatFormat.encode_dialog_prompt(dialog)

class ChatFormat:

    @staticmethod
    def encode_header( message: Message) -> List[str]:
        prompts = []
        prompts.append("<|start_header_id|>")
        prompts.extend(message["role"])
        prompts.append("<|end_header_id|>")
        prompts.extend("\n\n")
        return prompts

    @staticmethod
    def encode_message(message: Message) -> List[str]:
        prompts = ChatFormat.encode_header(message)
        prompts.extend(message["content"].strip())
        prompts.append("<|eot_id|>")
        return prompts

    @staticmethod
    def encode_dialog_prompt(dialog: Dialog) -> str:
        prompts = ["<|begin_of_text|>"]
        for message in dialog:
            prompts.extend(ChatFormat.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        prompts.extend(ChatFormat.encode_header({"role": "assistant", "content": ""}))
        return {
            prompt_output: "".join(prompts),
            python_output: "",
            hf_output: "",
        }


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("## Configurations")
            model = gr.Dropdown(["Llama 3", "Llama 2"], label="Model", filterable=False)

            prompt_template = gr.Dropdown(["Single Turn", "Multi Turn"], label="Prompt Template", filterable=False)

            gr.Markdown("## Input Prompts")
            system_prompt = gr.Textbox(label="System prompt", lines=2)
            user_prompt_1 = gr.Textbox(label="User prompt", lines=2)
            assistant_response = gr.Textbox(label="Assistant response", lines=2, visible=False, elem_id="assistant_response")
            user_prompt_2 = gr.Textbox(label="User prompt", lines=2, visible=False, elem_id="user_prompt_2")

            prompt_template.input(prompt_template_dropdown_listener, prompt_template, [assistant_response, user_prompt_2])


        with gr.Column(scale=3, min_width=600):
            gr.Markdown("## Output")
            with gr.Tab("Preview"):
                prompt_output = gr.Textbox(show_label=False, interactive=False, min_width=600, lines=30)

            with gr.Tab("Code"):
                with gr.Row():
                    with gr.Tab("Plain Python"):
                        python_output = gr.Textbox(label="Python Code", interactive=False, min_width=600, lines=25)
                    with gr.Tab("Hugging Face"):
                        hf_output = gr.Textbox(label="Using HF Transformers", interactive=False, min_width=600, lines=25)

    system_prompt.change(format_prompt_template_listener, [system_prompt, user_prompt_1, assistant_response, user_prompt_2], [prompt_output, python_output, hf_output])
    user_prompt_1.change(format_prompt_template_listener, [system_prompt, user_prompt_1, assistant_response, user_prompt_2], [prompt_output, python_output, hf_output])
    assistant_response.change(format_prompt_template_listener, [system_prompt, user_prompt_1, assistant_response, user_prompt_2], [prompt_output, python_output, hf_output])
    user_prompt_2.change(format_prompt_template_listener, [system_prompt, user_prompt_1, assistant_response, user_prompt_2], [prompt_output, python_output, hf_output])



demo.launch()
