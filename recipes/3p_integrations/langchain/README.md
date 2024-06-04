# LangChain <> Llama3 Cookbooks

### `Agents`

LLM agents use [planning, memory, and tools](https://lilianweng.github.io/posts/2023-06-23-agent/) to accomplish tasks. Here, we show how to build agents capable of [tool-calling](https://python.langchain.com/docs/integrations/chat/) using [LangGraph](https://python.langchain.com/docs/langgraph) with Llama 3. 

Agents can empower Llama 3 with important new capabilities. In particular, we will show how to give Llama 3 the ability to perform web search, as well as multi-modality: image generation (text-to-image), image analysis (image-to-text), and voice (text-to-speech) tools!

Tool-calling agents with LangGraph use two nodes: (1) a node with an LLM decides which tool to invoke based upon the user question. It outputs the tool name and arguments to use. (2) the tool name and arguments are passed to a tool node, which calls the tool itself with the specified arguments and returns the result back to the LLM.

![Screenshot 2024-05-30 at 10 48 58 AM](https://github.com/rlancemartin/llama-recipes/assets/122662504/a2c2ec40-2c7b-486e-9290-33b6da26c304)

Our first notebook, `langgraph-tool-calling-agent`, shows how to build our agent mentioned above using LangGraph.

See this [video overview](https://www.youtube.com/watch?v=j2OAeeujQ9M) for more detail on the design of this agent.

--- 

### `RAG Agent`

Our second notebook, `langgraph-rag-agent`, shows how to apply LangGraph to build a custom Llama 3 powered RAG agent that uses ideas from 3 papers:

* Corrective-RAG (CRAG) [paper](https://arxiv.org/pdf/2401.15884.pdf) uses self-grading on retrieved documents and web-search fallback if documents are not relevant.
* Self-RAG [paper](https://arxiv.org/abs/2310.11511) adds self-grading on generations for hallucinations and for ability to answer the question.
* Adaptive RAG [paper](https://arxiv.org/abs/2403.14403) routes queries between different RAG approaches based on their complexity.

We implement each approach as a control flow in LangGraph:
- **Planning:** The sequence of RAG steps (e.g., retrieval, grading, and generation) that we want the agent to take.
- **Memory:** All the RAG-related information (input question, retrieved documents, etc) that we want to pass between steps.
- **Tool use:** All the tools needed for RAG (e.g., decide web search or vectorstore retrieval based on the question).

We will build from CRAG (blue, below) to Self-RAG (green) and finally to Adaptive RAG (red):

![langgraph_rag_agent_](https://github.com/rlancemartin/llama-recipes/assets/122662504/ec4aa1cd-3c7e-4cd1-a1e7-7deddc4033a8)

--- 
 
### `Local LangGraph RAG Agent`

Our third notebook, `langgraph-rag-agent-local`, shows how to apply LangGraph to build advanced RAG agents using Llama 3 that run locally and reliably.

See this [video overview](https://www.youtube.com/watch?v=sgnrL7yo1TE) for more detail on the design of this agent.
