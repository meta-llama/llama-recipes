## [VideoSummary](video_summary.ipynb): Ask Llama 3 to Summarize a Long YouTube Video (using Replicate or [OctoAI](../3p_integrations/octoai/video_summary.ipynb))
This demo app uses Llama 3 to return a text summary of a YouTube video. It shows how to retrieve the caption of a YouTube video and how to ask Llama to summarize the content in different ways, from the simplest naive way that works for short text to more advanced methods of using LangChain's map_reduce and refine to overcome the 8K context length limit of Llama 3.

## [NBA2023-24](./coding/text2sql/structured_llama.ipynb): Ask Llama 3 about Structured Data
This demo app shows how to use LangChain and Llama 3 to let users ask questions about **structured** data stored in a SQL DB. As the 2023-24 NBA season is entering the playoff, we use the NBA roster info saved in a SQLite DB to show you how to ask Llama 3 questions about your favorite teams or players.

## [live_data](live_data.ipynb): Ask Llama 3 about Live Data (using Replicate or [OctoAI](../3p_integrations/octoai/live_data.ipynb))
This demo app shows how to perform live data augmented generation tasks with Llama 3, [LlamaIndex](https://github.com/run-llama/llama_index), another leading open-source framework for building LLM apps, and the [Tavily](https://tavily.com) live search API.

## [WhatsApp Chatbot](./customerservice_chatbots/whatsapp_chatbot/whatsapp_llama3.md): Building a Llama 3 Enabled WhatsApp Chatbot
This step-by-step tutorial shows how to use the [WhatsApp Business API](https://developers.facebook.com/docs/whatsapp/cloud-api/overview) to build a Llama 3 enabled WhatsApp chatbot.

## [Messenger Chatbot](./customerservice_chatbots/messenger_chatbot/messenger_llama3.md): Building a Llama 3 Enabled Messenger Chatbot
This step-by-step tutorial shows how to use the [Messenger Platform](https://developers.facebook.com/docs/messenger-platform/overview) to build a Llama 3 enabled Messenger chatbot.

### RAG Chatbot Example (running [locally](./customerservice_chatbots/RAG_chatbot/RAG_Chatbot_Example.ipynb) or on [OctoAI](../3p_integrations/octoai/RAG_chatbot_example/RAG_chatbot_example.ipynb))
A complete example of how to build a Llama 3 chatbot hosted on your browser that can answer questions based on your own data using retrieval augmented generation (RAG). You can run Llama2 locally if you have a good enough GPU or on OctoAI if you follow the note [here](../README.md#octoai_note).

## [Sales Bot](./customerservice_chatbots/ai_agent_chatbot/SalesBot.ipynb): Sales Bot with Llama3 - A Summarization and RAG Use Case
An summarization + RAG use case built around the Amazon product review Kaggle dataset to build a helpful Music Store Sales Bot. The summarization and RAG are built on top of Llama models hosted on OctoAI, and the vector database is hosted on Weaviate Cloud Services.

## [Media Generation](./MediaGen.ipynb): Building a Video Generation Pipeline with Llama3
This step-by-step tutorial shows how to use leverage Llama 3 to drive the generation of animated videos using SDXL and SVD. More specifically it relies on JSON formatting to produce a scene-by-scene story board of a recipe video. The user provides the name of a dish, then Llama 3 describes a step by step guide to reproduce the said dish. This step by step guide is brought to life with models like SDXL and SVD.
