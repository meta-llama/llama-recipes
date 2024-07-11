# Building Agentic RAG with Llamaindex

The folder here containts the Llama 3 ported notebooks of the DLAI short course [Building Agentic RAG with Llamaindex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/).

1. [Building Agentic RAG with Llamaindex L1 Router Engine](../../../quickstart/agents/dlai/Building_Agentic_RAG_with_Llamaindex_L1_Router_Engine.ipynb) shows how to implement a simple agentic RAG, a router that will pick up one of several query tools (question answering or summarization) to execute a query on a single document. Note this notebook is located in the `quickstart` folder.

2. [Building Agentic RAG with Llamaindex L2 Tool Calling](Building_Agentic_RAG_with_Llamaindex_L2_Tool_Calling.ipynb) shows how to use Llama 3 to not only pick a function to execute, but also infer an argument to pass through the function.

3. [Building Agentic RAG with Llamaindex L3 Building an Agent Reasoning Loop](Building_Agentic_RAG_with_Llamaindex_L3_Building_an_Agent_Reasoning_Loop.ipynb) shows how to define a complete agent reasoning loop to reason over tools and multiple steps on a complex question the user asks about a single document while maintaining memory.

3. [Building Agentic RAG with Llamaindex L4 Building a Multi-Document Agent](Building_Agentic_RAG_with_Llamaindex_L4_Building_a_Multi-Document_Agent.ipynb) shows how to use an agent to handle multiple documents and increasing degrees of complexity.