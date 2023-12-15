# Inference Throughput Benchmarks
In this folder we provide a series of benchmark scripts that apply a throughput analysis for Llama 2 models inference on various backends:
* On-perm - Popular serving framework and containers (i.e. vLLM)
* [**WIP**]Cloud API - Popular API services (i.e. Azure Model-as-a-Service)
* [**WIP**]On-device - Popular on-device inference solutions on Android and iOS (i.e. mlc-llm, QNN)
* [**WIP**]Optimization - Popular optimization solutions for faster inference and quantization (i.e. AutoAWQ)

# Why
There are three major reasons we want to run these benchmarks and share them with our Llama community:
* Provide inference throughput analysis based on real world situation to help you better select which service or deployment works the best for your scenario
* Provide a baseline measurement for validating various optimization solutions on different backends, so we can provide guidance on which solutions works the best for your scenario
* Encourge the community to develop benchmarks on top of our works, so we can better quantify the latest proposed solutions combined with current popular frameworks, especially in this crazy fast-moving area

# Parameters
Here are the parameters (if applicable) that you can configure for running the benchmark:
* **PROMPT** - Prompt sent in for inference (configure the length of prompt, choose from 5, 25, 50, 100, 500, 1k and 2k)
* **MAX_NEW_TOKEN** - Max token generated
* **CONCURRENT_LEVELS** - Max number of concurrent requests
* **MODEL_PATH** - Model source
* **MODEL_HEADERS** - Request headers
* **SAFE_CHECK** - Content safety check (either Azure service or simulated latency)
* **THRESHOLD_TPS** - Threshold TPS (threshold for tokens per second below which we deem the query to be slow)
* **TOKENIZER_PATH** - Tokenizer source
* **RANDOM_PROMPT_LENGTH** - Random prompt length (for pretrained models)
* **NUM_GPU** - Number of GPUs for request dispatch among muiltiple containers
* **TEMPERATURE** - Temperature for inference
* **TOP_P** - Top_p for inference
* **MODEL_ENDPOINTS** - Container endpoints
* Model parallelism or model replicas

You can also configure other model hyperparameters as part of the request payload.  
All these parameters are stored in ```parameter.json``` and real prompts are stored in ```input.jsonl```. Running the script will load these configurations.



# Metrics
The benchmark will report these metrics per instance:
* Number of concurrent requests
* P50 Latency(ms)
* P99 Latency(ms)
* Request per second (RPS)
* Output tokens per second
* Output tokens per second per GPU
* Input tokens per second
* Input tokens per second per GPU
* Average tokens per second per request

We intend to add these metrics in the future:
* Time to first token (TTFT)
  
The benchmark result will be displayed in terminal output and saved as a CSV file (```performance_metrics.csv```) that you can export to spreadsheets.

# Getting Started
Please follow the ```README.md``` in each subfolder for instructions on how to setup and run these benchmarks. 

