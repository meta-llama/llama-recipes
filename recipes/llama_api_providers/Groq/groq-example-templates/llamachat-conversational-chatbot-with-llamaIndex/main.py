from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage

llm = Groq(model="llama3-8b-8192")


system_prompt = 'You are a friendly but highly sarcastic chatbot assistant'

while True:
    # Get the user's question
    user_input = input("User: ")

    #user_input = 'write a few paragraphs explaining generative AI to a college freshman'

    ##################################
    # Simple Chat
    ##################################
    print('Simple Chat:\n\n')
    response = llm.complete(user_input)
    print(response)


    ##################################
    # Streaming Chat
    ##################################
    stream_response = llm.stream_complete(
        user_input
    )
    print('\n\nStreaming Chat:\n')
    for t in stream_response:
        print(t.delta, end="")


    ##################################
    # Customizable Chat
    ##################################
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_input),
    ]
    print('\n\nChat with System Prompt:\n')
    response_with_system_prompt = llm.chat(messages)

    print(response_with_system_prompt)


