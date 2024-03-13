## VideoSummary: Ask Llama2 to Summarize a YouTube Video (using [Replicate](VideoSummary.ipynb) or [OctoAI](../llama_api_providers/OctoAI_API_examples/VideoSummary.ipynb))
This demo app uses Llama2 to return a text summary of a YouTube video. It shows how to retrieve the caption of a YouTube video and how to ask Llama to summarize the content in four different ways, from the simplest naive way that works for short text to more advanced methods of using LangChain's map_reduce and refine to overcome the 4096 limit of Llama's max input token size.

## [NBA2023-24](./text2sql/StructuredLlama.ipynb): Ask Llama2 about Structured Data
This demo app shows how to use LangChain and Llama2 to let users ask questions about **structured** data stored in a SQL DB. As the 2023-24 NBA season is around the corner, we use the NBA roster info saved in a SQLite DB to show you how to ask Llama2 questions about your favorite teams or players.

## LiveData: Ask Llama2 about Live Data (using [Replicate](LiveData.ipynb) or [OctoAI](../llama_api_providers/OctoAI_API_examples/LiveData.ipynb))
This demo app shows how to perform live data augmented generation tasks with Llama2 and [LlamaIndex](https://github.com/run-llama/llama_index), another leading open-source framework for building LLM apps: it uses the [You.com search API](https://documentation.you.com/quickstart) to get live search result and ask Llama2 about them.

## [WhatsApp Chatbot](./chatbots/whatsapp_llama/whatsapp_llama2.md): Building a Llama-enabled WhatsApp Chatbot
This step-by-step tutorial shows how to use the [WhatsApp Business API](https://developers.facebook.com/docs/whatsapp/cloud-api/overview) to build a Llama-enabled WhatsApp chatbot.

## [Messenger Chatbot](./chatbots/messenger_llama/messenger_llama2.md): Building a Llama-enabled Messenger Chatbot
This step-by-step tutorial shows how to use the [Messenger Platform](https://developers.facebook.com/docs/messenger-platform/overview) to build a Llama-enabled Messenger chatbot.

### RAG Chatbot Example (running [locally](./chatbots/RAG_chatbot/RAG_Chatbot_Example.ipynb) or on [OctoAI](../llama_api_providers/OctoAI_API_examples/RAG_Chatbot_example/RAG_Chatbot_Example.ipynb))
A complete example of how to build a Llama 2 chatbot hosted on your browser that can answer questions based on your own data using retrieval augmented generation (RAG). You can run Llama2 locally if you have a good enough GPU or on OctoAI if you follow the note [here](../README.md#octoai_note).