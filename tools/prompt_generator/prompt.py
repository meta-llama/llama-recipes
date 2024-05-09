import gradio as gr

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

        return {prompt_output: f"""
<|begin_text|>{system_prompt}
{user_prompt_1}
""",
        python_output: f"""
        """,
        hf_output: f"""
        """,
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



demo.launch()
