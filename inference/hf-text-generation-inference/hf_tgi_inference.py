import json
from text_generation import Client


def build_chat_completion_prompt(dialog: list[str]) -> str:
    """
    Builds a chat completion prompt from a dialog of System, User, and Assistant messages.
    ref: https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/model.py

    The function formats the conversation according to the below schema, including system prompts,
    user messages, and assistant messages. If there is a system prompt, it is enclosed within <<SYS>> tags.
    User and assistant messages are enclosed within [INST] and [/INST] tags.

    .. code-block:: text
        <s>[INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]


    :param dialog: A list of dictionaries containing the dialog. Each dictionary must have a "role" key
                   (with values "system", "user", or "assistant") and a "content" key with the message content.
    :type dialog: list[dict[str, str]]

    :return: A formatted string representing the conversation.
    :rtype: str

    Example:

    .. code-block:: python

        dialog = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Good Morning!"},
            {"role": "assistant", "content": "Good morning! How can I help you today?"},
        ]
        formatted_conversation = build_chat_completion_prompt(dialog)
    """

    texts = ["<s>[INST]"]  # Start the conversation with <s>[INST]
    do_strip = False  # Tracks whether to strip the leading and trailing spaces from the user input
    user_message = None  # Holds the current user message being processed

    # Iterate through the dialog to format the conversation
    for i, message in enumerate(dialog):
        role = message["role"]
        content = message["content"]

        if role == "system" and i == 0:  # Add system prompt
            texts.append(f"<<SYS>>\n{content}\n<</SYS>>\n\n")
        elif role == "user":
            user_message = content.strip() if do_strip else content
            do_strip = True
        elif role == "assistant" and user_message is not None:
            texts.append(f"{user_message} [/INST] {content.strip()} </s><s>[INST] ")
            user_message = None

    # Handle the last user message, if there is one
    if user_message is not None:
        texts.append(f"{user_message} [/INST]")

    return "".join(texts)


# Create a text-generation client to interact with the model
client = Client("http://127.0.0.1:8080")

# Load the sample dialogs
with open("../chats.json", "r") as f:
    sample_dialogs = json.load(f)

# Generate responses for the sample dialogs
for dialog in sample_dialogs:
    prompt = build_chat_completion_prompt(dialog)
    print(client.generate(prompt, max_new_tokens=200).generated_text.strip())
    print("-" * 80)
