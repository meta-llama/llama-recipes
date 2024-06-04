# LangChain <> Llama3 Cookbooks

LLM agents use [planning, memory, and tools](https://lilianweng.github.io/posts/2023-06-23-agent/) to accomplish tasks. Agents can empower Llama 3 with important new capabilities. Here, we will show how to give Llama 3 the ability to perform web search, as well as multi-modality: image generation (text-to-image), image analysis (image-to-text), and voice (text-to-speech) tools!

LangChain offers several different ways to implement agents with Llama 3:

(1) `ReAct agent` - Uses [AgentExecutor](https://python.langchain.com/docs/modules/agents/quick_start/) with [tool-calling](https://python.langchain.com/docs/integrations/chat/) versions of Llama 3.

(2) `LangGraph tool calling agent` - Uses [LangGraph](https://python.langchain.com/docs/langgraph) with [tool-calling](https://python.langchain.com/docs/integrations/chat/) versions of Llama 3.

(3) `LangGraph custom agent` - Uses [LangGraph](https://python.langchain.com/docs/langgraph) with **any** version of Llama 3 (so long as it supports structured output).

As we move from option (1) to (3) the degree of customization and flexibility increases:

(1) `ReAct agent` using AgentExecutor is a great for getting started quickly with minimal code, but requires a version of Llama 3 with reliable tool-calling, is the least customizable, and uses higher-level AgentExecutor abstraction.
  
(2) `LangGraph tool calling agent` is more customizable than (1) because the LLM assistant (planning) and tool call (action) nodes are defined by the user, but it still requires a version of Llama 3 with reliable tool-calling.
  
(3) `LangGraph custom agent` does not require a version of Llama 3 with reliable tool-calling and is the most customizable, but requires the most work to implement. 

![langgraph_agent_architectures](https://github.com/rlancemartin/llama-recipes/assets/122662504/5ed2bef0-ae11-4efa-9e88-ab560a4d0022)

---

### `ReAct agent`

The AgentExecutor manages the loop of planning, executing tool calls, and processing outputs until an AgentFinish signal is generated, indicating task completion.

Our first notebook, `tool-calling-agent`, shows how to build a [tool calling agent](https://python.langchain.com/docs/modules/agents/agent_types/tool_calling/) with AgentExecutor and Llama 3.

--- 

### `LangGraph tool calling agent`

[LangGraph](https://python.langchain.com/docs/langgraph) is a library from LangChain that can be used to build reliable agents.

Our second notebook, `langgraph-tool-calling-agent`, shows an alternative to AgentExecutor for building a Llama 3 powered agent. 

--- 

### `LangGraph custom agent`

Our third notebook, `langgraph-custom-agent`, shows how to build a Llama 3 powered agent without reliance on tool-calling. 

--- 

### `LangGraph RAG Agent`

Our fourth notebook, `langgraph-rag-agent`, shows how to apply LangGraph to build a custom Llama 3 powered RAG agent that use ideas from 3 papers:

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

Our fifth notebook, `langgraph-rag-agent-local`, shows how to apply LangGraph to build advanced RAG agents using Llama 3 that run locally and reliably.

See this [video overview](https://www.youtube.com/watch?v=sgnrL7yo1TE) for more detail on the design of this agent.
