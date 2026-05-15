# Using Llama on Astraflow

[Astraflow](https://astraflow.ucloud-global.com) by UCloud (优刻得) is an OpenAI-compatible AI model aggregation platform that provides access to 200+ models — including Meta Llama — through a single, unified API.

Because Astraflow exposes an OpenAI-compatible endpoint, you can use the standard `openai` Python SDK and simply point it at Astraflow's base URL.

## Endpoints

| Region | Base URL | Environment variable |
|--------|----------|----------------------|
| Global | `https://api-us-ca.umodelverse.ai/v1` | `ASTRAFLOW_API_KEY` |
| China  | `https://api.modelverse.cn/v1`        | `ASTRAFLOW_CN_API_KEY` |

## Prerequisites

1. Sign up for an API key:
   - **Global**: https://astraflow.ucloud-global.com
   - **China**: https://astraflow.ucloud.cn
2. Install the OpenAI Python SDK:
   ```bash
   pip install openai
   ```
3. Export your key:
   ```bash
   export ASTRAFLOW_API_KEY="your-api-key-here"
   ```

## Quick Start

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["ASTRAFLOW_API_KEY"],
    base_url="https://api-us-ca.umodelverse.ai/v1",
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What can you tell me about Meta Llama models?"},
    ],
)
print(response.choices[0].message.content)
```

## Cookbooks

| Cookbook | Description |
|----------|-------------|
| [Llama on Astraflow — Getting Started](./llama_on_astraflow.ipynb) | Chat completions and streaming with Llama models via the Astraflow OpenAI-compatible API |

## Additional Resources

- [Astraflow Documentation (Global)](https://astraflow.ucloud-global.com)
- [Astraflow Documentation (China)](https://astraflow.ucloud.cn)
- [UCloud Official Website](https://www.ucloud.cn)
