# LangChain <> Llama3 Cookbooks

LLM agents use [planning, memory, and tools](https://lilianweng.github.io/posts/2023-06-23-agent/) to accomplish tasks.

LangChain offers several different ways to implement agents.

(1) Use [AgentExecutor](https://python.langchain.com/docs/modules/agents/quick_start/) with [tool-calling](https://python.langchain.com/docs/integrations/chat/) versions of Llama 3.

(2) Use [LangGraph](https://python.langchain.com/docs/langgraph), a library from LangChain that can be used to build reliable agents with Llama 3.

---

### AgentExecutor Agent

AgentExecutor is the runtime for an agent. AgentExecutor calls the agent, executes the actions it chooses, passes the action outputs back to the agent, and repeats.

Our first notebook, `tool-calling-agent`, shows how to build a [tool calling agent](https://python.langchain.com/docs/modules/agents/agent_types/tool_calling/) with AgentExecutor and Llama 3.

This shows how to build an agent that uses web search and retrieval tools.

--- 

### LangGraph Agent

[LangGraph](https://python.langchain.com/docs/langgraph) is a library from LangChain that can be used to build reliable agents.

LangGraph can be used to build agents with a few pieces:
- **Planning:** Define a control flow of steps that you want the agent to take (a graph)
- **Memory:** Persist information (graph state) across these steps
- **Tool use:** Modify state at any step

Our second notebook, `langgraph-agent`, shows how to build a Llama 3 powered agent that uses web search and retrieval tool in LangGraph.

It discusses some of the trade-offs between AgentExecutor and LangGraph.

--- 

### LangGraph RAG Agent

Our third notebook, `langgraph-rag-agent`, shows how to apply LangGraph to build advanced Llama 3 powered RAG agents that use ideas from 3 papers:

* Corrective-RAG (CRAG) [paper](https://arxiv.org/pdf/2401.15884.pdf) uses self-grading on retrieved documents and web-search fallback if documents are not relevant.
* Self-RAG [paper](https://arxiv.org/abs/2310.11511) adds self-grading on generations for hallucinations and for ability to answer the question.
* Adaptive RAG [paper](https://arxiv.org/abs/2403.14403) routes queries between different RAG approaches based on their complexity.

We implement each approach as a control flow in LangGraph:
- **Planning:** The sequence of RAG steps (e.g., retrieval, grading, and generation) that we want the agent to take
- **Memory:** All the RAG-related information (input question, retrieved documents, etc) that we want to pass between steps
- **Tool use:** All the tools needed for RAG (e.g., decide web search or vectorstore retrieval based on the question)

We will build from CRAG (blue, below) to Self-RAG (green) and finally to Adaptive RAG (red):

![Screenshot 2024-05-03 at 10 50 02 AM](https://github.com/rlancemartin/llama-recipes/assets/122662504/ec4aa1cd-3c7e-4cd1-a1e7-7deddc4033a8)

--- 

### Local LangGraph RAG Agent

Our fourth notebook, `langgraph-rag-agent-local`, shows how to apply LangGraph to build advanced RAG agents using Llama 3 that run locally and reliably.

See this [video overview](https://www.youtube.com/watch?v=sgnrL7yo1TE) for more detail.
