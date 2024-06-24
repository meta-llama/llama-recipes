# Tune Llama 3 for text-to-SQL and improve accuracy from 30% to 95%

This repo and notebook `meta-lamini.ipynb` demonstrate how to tune Llama 3 to generate valid SQL queries and improve accuracy from 30% to 95%.

In this notebook we'll be using Lamini, and more specifically, Lamini Memory Tuning. 

Lamini is an integrated platform for LLM inference and tuning for the enterprise. Lamini Memory Tuning is a new tool you can use to embed facts into LLMs that improves factual accuracy and reduces hallucinations. Inspired by information retrieval, this method has set a new standard of accuracy for LLMs with less developer effort. 

Learn more about Lamini Memory Tuning: https://www.lamini.ai/blog/lamini-memory-tuning

Please head over to https://app.lamini.ai/account to get your free api key.

You can authenticate by writing the following to a file `~/.lamini/configure.yaml`

```
production:
    key: <YOUR-LAMINI-API-KEY>
```

This tuning tutorial uses the `nba_roster` sqlite database to tune a Llama 3 model.

## Additional resources

▫️ Fortune 500 case study: http://www.lamini.ai/blog/llm-text-to-sql <br>
▫️ Technical paper: https://github.com/lamini-ai/Lamini-Memory-Tuning/blob/main/research-paper.pdf <br>
▫️ Model weights: https://huggingface.co/engineering-lamini/lamini-1-random
