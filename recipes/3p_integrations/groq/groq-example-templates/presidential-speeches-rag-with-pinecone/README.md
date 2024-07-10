# Presidential Speeches RAG with Pinecone

This repository contains a command line application that allows users to ask questions about US presidental speeches by applying Retrieval-Augmented Generation (RAG) over a Pinecone vector database. The application uses RAG to answer the user's question by retrieving the most relevant presidential speeches and using them to supplant the LLM response.

## Features

- **RAG (Retrieval-Augmented Generation)**: Enhances the generation of responses by integrating retrieval-based methods. This feature allows the system to fetch relevant information from a large corpus of data, providing more accurate and contextually appropriate answers by combining retrieved content with generative capabilities.

- **Vector Databases (Pinecone)**: Integrates with Pinecone to store and manage vector embeddings efficiently. Pinecone's high-performance vector database allows for fast and scalable similarity searches, enabling quick retrieval of relevant data for various machine learning and AI applications.

- **LangChain Integration**: Leverages LangChain to facilitate natural language processing tasks. LangChain enhances the interaction between the user and the system by providing robust language modeling capabilities, ensuring seamless and intuitive communication.

## Code Overview

The main script of the application is [main.py](./main.py). Here's a brief overview of its main functions:

- `get_relevant_excerpts(user_question, docsearch)`: This function takes a user's question and a Pinecone vector store as input, performs a similarity search on the vector store using the user's question, and returns the most relevant excerpts from presidential speeches.

- `get_relevant_excerpts(user_question, docsearch)`: This function takes a user's question and a Pinecone vector store as input, performs a similarity search on the vector store using the user's question, and returns the most relevant excerpts from presidential speeches.

- `presidential_speech_chat_completion(client, model, user_question, relevant_excerpts, additional_context)`: This function takes a Groq client, a pre-trained model, a user's question, relevant excerpts from presidential speeches, and additional context as input. It generates a response to the user's question based on the relevant excerpts and the additional context

## Usage

<!-- markdown-link-check-disable -->

You will need to store a valid Groq API Key as a secret to proceed with this example outside of this Repl. You can generate one for free [here](https://console.groq.com/keys).

<!-- markdown-link-check-enable -->

You would also need your own [Pinecone](https://www.pinecone.io/) index with presidential speech embeddings to run this code locally. You can create a Pinecone API key and one index for a small project for free on their Starter plan, and visit [this Cookbook post](https://github.com/groq/groq-api-cookbook/blob/dan/replit-conversion/presidential-speeches-rag/presidential-speeches-rag.ipynb) for more info on RAG and a guide to uploading these embeddings to a vector database

You can [fork and run this application on Replit](https://replit.com/@GroqCloud/Presidential-Speeches-RAG-with-Pinecone) or run it on the command line with `python main.py`.
