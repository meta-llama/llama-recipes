import pandas as pd
import numpy as np
from groq import Groq
from pinecone import Pinecone
import os

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore


def get_relevant_excerpts(user_question, docsearch):
    """
    This function retrieves the most relevant excerpts from presidential speeches based on the user's question.
    Parameters:
    user_question (str): The question asked by the user.
    docsearch (PineconeVectorStore): The Pinecone vector store containing the presidential speeches.
    Returns:
    str: A string containing the most relevant excerpts from presidential speeches.
    """

    # Perform a similarity search on the Pinecone vector store using the user's question
    relevent_docs = docsearch.similarity_search(user_question)

    # Extract the page content from the top 3 most relevant documents and join them into a single string
    relevant_excerpts = '\n\n------------------------------------------------------\n\n'.join([doc.page_content for doc in relevent_docs[:3]])

    return relevant_excerpts


def presidential_speech_chat_completion(client, model, user_question, relevant_excerpts):
    """
    This function generates a response to the user's question using a pre-trained model.
    Parameters:
    client (Groq): The Groq client used to interact with the pre-trained model.
    model (str): The name of the pre-trained model.
    user_question (str): The question asked by the user.
    relevant_excerpts (str): A string containing the most relevant excerpts from presidential speeches.
    Returns:
    str: A string containing the response to the user's question.
    """

    # Define the system prompt
    system_prompt = '''
    You are a presidential historian. Given the user's question and relevant excerpts from 
    presidential speeches, answer the question by including direct quotes from presidential speeches. 
    When using a quote, site the speech that it was from (ignoring the chunk).
    '''

    # Generate a response to the user's question using the pre-trained model
    chat_completion = client.chat.completions.create(
        messages = [
            {
                "role": "system",
                "content":  system_prompt
            },
            {
                "role": "user",
                "content": "User Question: " + user_question + "\n\nRelevant Speech Exerpt(s):\n\n" + relevant_excerpts,
            }
        ],
        model = model
    )

    # Extract the response from the chat completion
    response = chat_completion.choices[0].message.content

    return response


def main():
    """
    This is the main function that runs the application. It initializes the Groq client and the SentenceTransformer model,
    gets user input from the Streamlit interface, retrieves relevant excerpts from presidential speeches based on the user's question,
    generates a response to the user's question using a pre-trained model, and displays the response.
    """

    model = 'llama3-8b-8192'

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize the Groq client
    groq_api_key = os.getenv('GROQ_API_KEY')
    pinecone_api_key=os.getenv('PINECONE_API_KEY')
    pinecone_index_name = "presidential-speeches"
    client = Groq(
        api_key=groq_api_key
    )

    pc = Pinecone(api_key = pinecone_api_key)
    docsearch = PineconeVectorStore(index_name=pinecone_index_name, embedding=embedding_function)

    # Display the title and introduction of the application
    print("Presidential Speeches RAG")
    multiline_text = """
    Welcome! Ask questions about U.S. presidents, like "What were George Washington's views on democracy?" or "What did Abraham Lincoln say about national unity?". The app matches your question to relevant excerpts from presidential speeches and generates a response using a pre-trained model.
    """

    print(multiline_text)


    while True:
        # Get the user's question
        user_question = input("Ask a question about a US president: ")

        if user_question:
            pinecone_index_name = "presidential-speeches"
            relevant_excerpts = get_relevant_excerpts(user_question, docsearch)
            response = presidential_speech_chat_completion(client, model, user_question, relevant_excerpts)
            print(response)



if __name__ == "__main__":
    main()