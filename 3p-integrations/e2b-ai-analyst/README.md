# AI Analyst with Llama and E2B
This is an AI-powered code and data analysis tool powered by Meta Llama and the [E2B SDK](https://e2b.dev/docs).

→ Try on [ai-analyst.e2b.dev](https://ai-analyst.e2b.dev/)

## Features
- 🔸 Analyze data with Meta's Llama 3.1 and 3.2
- 🔸 Upload CSV files
- 🔸 Create interactive charts

**Powered by:**

- 🔸 ✶ [E2B Sandbox](https://github.com/e2b-dev/code-interpreter)
- 🔸 Vercel's AI SDK
- 🔸 Next.js
- 🔸 echarts library for interactive charts

**Supported LLM Providers:**
- 🔸 TogetherAI
- 🔸 Fireworks
- 🔸 Ollama

**Supported chart types:**
- 🔸 All the supported charts are described [here](https://e2b.dev/docs/code-interpreting/create-charts-visualizations/interactive-charts#supported-intertactive-charts).


## Get started

Visit the [online version](https://ai-analyst.e2b.dev/) or run locally on your own.

### 1. Clone repository
```
git clone https://github.com/e2b-dev/ai-analyst.git
```

### 2. Install dependencies
```
cd ai-analyst && npm i
```

### 3. Add E2B API key
Copy `.example.env` to `.env.local` and fill in `E2B_API_KEY`.

- Get your [E2B API key here](https://e2b.dev/dashboard?tab=keys).

### 4. Configure LLM provider

In `.env.local`, add an API key for at least one LLM provider:

- Fireworks: `FIREWORKS_API_KEY`
- Together AI: `TOGETHER_API_KEY`

For Ollama, provide the base URL instead of the API key:

- Ollama: `OLLAMA_BASE_URL`
