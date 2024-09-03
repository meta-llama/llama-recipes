# Using Llama3.1 built in tools
Meta's latest Llama3.1 models offer unique function calling capabilities. In particular they offer built-in tool calling capabilities for the following 3 external tools:
* Brave Search: internet search
* Code Interpreter: Python code interpreter
* Wolfram Alpha: mathematical and scientific knowledge tool

To sell the benefits of the built in tools, let's look at what one would get back from an LLM with or without tool calling capabilities. In particular:

### Code Interpreter
User Query: `I just got a 25 year mortgage of 400k at a fixed rate of 5.14% and paid 20% down. How much will I pay in interest?`
* Answer without tool calls (wrong): `Total paid interest: $184,471`
* Answer with tool calls (correct): `you will pay a total of $249,064.70 in interest`

### Brave Search
User Query: `What caused a wordlwide outage in the airlines industry in July of 2024?`
* Answer without tool calls: `I'm not aware of anything that would have caused a worldwide outage in the airlines industry.`
* Answer with tool calls: `The global technology outage was caused by a faulty software update that affected Windows programs running cybersecurity technology from CrowdStrike. The outage disrupted flights, media outlets, hospitals, small businesses, and government offices, highlighting the vulnerability of the world's interconnected systems.`

### Wolfram Alpha
User Query: `Derive the prime factorization of 892041`
* Answer without tool calls (wrong): `The prime factorization of 892041 is:\n\n2 × 2 × 2 × 3 × 3 × 3 × 5 × 13 × 17 × 17`
* Answer with tool calls (correct): `The prime factorization of 892041 is 3 × 17 × 17491.`

## What you will build
You will learn how to make use of these built in capabilities to address some of the notorious weaknesses of LLMs:
* Limited ability to reason about complex mathematical notions
* Limited ability to answer questions about current events (or data that wasn't included in the model's training set)

## What you will use
You'll learn to invoke Llama3.1 models hosted on OctoAI, and make use of the model's built in tool calling capabilities via a standardized OpenAI-compatible chat completions API.

## Instructions
Make sure you have Jupyter Notebook installed in your environment before launching the notebook in the `recipes/use_cases/tool_calling/builtins` directory.

The rest of the instructions are described in the notebook itself.
