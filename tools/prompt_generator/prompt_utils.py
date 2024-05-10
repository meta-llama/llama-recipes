from typing import (
    List,
    Literal,
    Sequence,
    TypedDict,
)

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]

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
        return "".join(prompts)
