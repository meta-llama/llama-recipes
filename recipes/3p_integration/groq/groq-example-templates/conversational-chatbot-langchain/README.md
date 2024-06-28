# Groq LangChain Conversational Chatbot

A simple application that allows users to interact with a conversational chatbot powered by LangChain. The application uses the Groq API to generate responses and leverages LangChain's [ConversationBufferWindowMemory](https://python.langchain.com/v0.1/docs/modules/memory/types/buffer_window/) to maintain a history of the conversation to provide context for the chatbot's responses.

## Features

- **Conversational Interface**: The application provides a conversational interface where users can ask questions or make statements, and the chatbot responds accordingly.

- **Contextual Responses**: The application maintains a history of the conversation, which is used to provide context for the chatbot's responses.

- **LangChain Integration**: The chatbot is powered by the LangChain API, which uses advanced natural language processing techniques to generate human-like responses.

## Usage

<!-- markdown-link-check-disable -->

You will need to store a valid Groq API Key as a secret to proceed with this example. You can generate one for free [here](https://console.groq.com/keys).

<!-- markdown-link-check-enable -->

You can [fork and run this application on Replit](https://replit.com/@GroqCloud/Chatbot-with-Conversational-Memory-on-LangChain) or run it on the command line with `python main.py`
