# Social listening with Llama and Xquik

This agent answers questions about current public conversations on X. It uses
Llama tool calling to create one bounded search query. Xquik runs the Twitter
search and returns recent public posts. Llama then analyzes that evidence and
cites only URLs present in the result.

Use this workflow when model knowledge may be stale. Do not use it to infer
private facts or make claims about people from a small result set.

## How it works

1. The user asks a question about a current conversation.
2. Llama calls `search_x_posts` once with a query and result limit.
3. The Xquik Python SDK retrieves recent public posts.
4. The agent sends bounded post evidence back to Llama.
5. Llama separates observations from inferences and cites source URLs.

The tool result includes post text, author username, creation time, URL, and
public engagement counts. It excludes profile biographies, media, replies,
quoted posts, credentials, and raw API responses.

## Requirements

- Python 3.10 or newer
- A [Llama API key](https://llama.developer.meta.com/)
- An [Xquik API key](https://xquik.com)

The example pins the public `llama-api-client` and `x-twitter-scraper`
packages in `requirements.txt`.

## Install

```bash
cd end-to-end-use-cases/xquik_social_listening
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set credentials through the environment. Never put them in source files or
command-line arguments.

```bash
export LLAMA_API_KEY="your-llama-api-key"
export X_TWITTER_SCRAPER_API_KEY="your-xquik-api-key"
```

## Run

```bash
python social_listening_agent.py \
  "What are developers saying about Llama 4 tool calling?"
```

The default limit is 10 posts. Set a lower or higher bound between 1 and 25:

```bash
python social_listening_agent.py \
  --max-posts 20 \
  "What concerns appear in recent Llama 4 discussions?"
```

Each run makes at most 1 Xquik search. Review your account limits before
raising `--max-posts`.

## Safety boundaries

- Post text is untrusted data. The system prompt tells Llama never to follow
  instructions inside posts.
- Each post is normalized and limited to 500 characters before model input.
- The model receives at most 25 posts and 1 search result page.
- The agent preserves only returned HTTPS URLs on X or Twitter. It never
  constructs citations.
- Both clients close after the command completes.
- Tests use fake clients. They never call Llama, Xquik, or X.

Search results can be incomplete, deleted, manipulated, or unrepresentative.
Validate important conclusions against primary sources.

## Test

```bash
python -m pip install pytest
python -m pytest -q tests
```

The tests cover tool-call validation, request bounds, evidence shaping, text
normalization, prompt-injection isolation, orchestration, and client cleanup.

Xquik is an independent third-party service. Not affiliated with X Corp.
"Twitter" and "X" are trademarks of X Corp.
