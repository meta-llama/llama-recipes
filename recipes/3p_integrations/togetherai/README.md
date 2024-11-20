# Building LLM apps using Llama on Together.ai

This folder contains demos on how to use Llama on [Together.ai](https://www.together.ai/) to quickly build LLM apps.

The first demo is a notebook that converts PDF to podcast using Llama 3.1 70B or 8B hosted by Together.ai. It differs and complements the [Meta's implementation](https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/NotebookLlama) in several ways:

1. You don't need to download the Llama models from HuggingFace and have a GPU to run the notebooks - you can quickly get a free Together API key and run the whole Colab notebook on a browser, in about 10 minutes;
2. A single system prompt is used to generate the naturally sounding podcast from PDF, with the support of pydantic, scratchpad and JSON response format to make the whole flow simple yet powerful;
3. A different TTS service, also with an easy-to-get free API key, is used.

The whole Colab notebook can run with a single "Runtime - Run all" click, generating the podcast audio from the Transformer paper that started the GenAI revolution. 

 
