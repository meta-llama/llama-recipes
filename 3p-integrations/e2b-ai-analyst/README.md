# AI Analyst with Llama and E2B
This is an AI-powered code and data analysis tool powered by Meta Llama and the [E2B SDK](https://e2b.dev/docs).

![Preview](https://private-user-images.githubusercontent.com/33395784/382129362-3bc8b017-4a09-416c-b55c-ce53da7e5560.gif?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzEwNzg5NDQsIm5iZiI6MTczMTA3ODY0NCwicGF0aCI6Ii8zMzM5NTc4NC8zODIxMjkzNjItM2JjOGIwMTctNGEwOS00MTZjLWI1NWMtY2U1M2RhN2U1NTYwLmdpZj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDExMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMTA4VDE1MTA0NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWRlZjc4ZDhmNzY1ZjExODM4OGEzMzFiZGRjMWU4ZmQ0OWI5MjJmNTY5MGZkZDk4MmI1MWRiMzFhY2UxYjM0NjcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Kaab1leGbgBiQhfh6bV1VHTc8QfinP7ufqwmn7Ra57c)

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