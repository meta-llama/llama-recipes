from llama_recipes.inference.llm import  LLM

together_example = LLM("TOGETHER::togethercomputer/llama-2-7b-chat::access-token")
together_result = together_example.query(prompt="Why is the sky blue?")
