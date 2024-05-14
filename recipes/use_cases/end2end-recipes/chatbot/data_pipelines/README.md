## Data Preprocessing Steps

### Step 1 : Prepare related documents

Download all your desired docs in PDF, Text or Markdown format to "data" folder inside the data_pipelines folder.

In this case we have an example of [Getting started with Meta Llama](https://llama.meta.com/get-started/) and other llama related documents such Llama3, Purple Llama, Code Llama papers along with Llama FAQ. Ideally, we should have searched all Llama documents across the web and follow the procedure below on them but that would be very costly for the purpose of a tutorial, so we will stick to our limited documents here.

### Step 2 : Prepare data (Q&A pairs) for fine-tuning

To use Meta Llama 3 70B model for the question and answer (Q&A) pair datasets creation from the prepared documents, we can either use Meta Llama 3 70B APIs from LLM cloud providers or host local LLM server.

In this example, we use OctoAI API as a demo, and the APIs could be replaced by any other API from other providers.

**NOTE** The generated data by these APIs or the model needs to be vetted to make sure about the quality.

```bash
export OCTOAI_API_TOKEN="OCTOAI_API_TOKEN"
python generate_question_answers.py
```

**NOTE** You need to be aware of your RPM (requests per minute), TPM (tokens per minute) and TPD (tokens per day), limit on your account in case using any of model API providers. In our case we had to process each document at a time. Then merge all the Q&A `json` files to make our dataset. We aimed for a specific number of Q&A pairs per document anywhere between 50-100. This is experimental and totally depends on your documents, wealth of information in them and how you prefer to handle question, short or longer answers etc.

Alternatively we can use on prem solutions such as the [TGI](../../../examples/hf_text_generation_inference/) or [VLLM](../../../examples/vllm/). Here we will use the prompt in the [./config.yaml] to instruct the model on the expected format and rules for generating the Q&A pairs. In this example, we will show how to create a vllm openai compatible server that host Meta Llama 3 70B instruct locally and generate the Q&A pair datasets.

```bash
# Make sure VLLM has been installed
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server  --model meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size 2 --disable-log-requests --port 8000
```

**NOTE** Please make sure the port has not been used. Since Meta Llama3 70B instruct model requires at least 135GB GPU memory, we need to use multiple GPUs to host it in a tensor parallel way.

Once the server is ready, we can query the server given the port number 8000 in another terminal. Here, "-v" sets the port number and "-t" sets the total questions we want to generate.

```bash
python generate_question_answers.py -v 8000 -t 800
```

This python program will read all the documents inside of "data" folder and split the data into batches by the context window limit (8K for Meta Llama3 and 4K for Llama 2) and apply the chat template, defined in "config.yaml", to each batch. Then it will use each batch to query VLLM server and save the return answers into data.json after some post-process steps.

### Step 3: Run the training

Run distributed training with:
```bash
CUDA_VISIBLE_DEVICES=0,1  torchrun --nnodes 1 --nproc_per_node 2  recipes/finetuning/finetuning.py --use_peft --enable_fsdp --peft_method lora  --model_name meta-llama/Meta-Llama-3-8B-Instruct --output_dir chatbot-8b --num_epochs 10 --batch_size_training 4 --dataset "custom_dataset" -custom_dataset.test_split "test" --custom_dataset.file "recipes/finetuning/datasets/chatbot_dataset.py" --use-wandb  --run_validation True  --custom_dataset.data_path 'recipes/use_cases/end2end-recipes/chatbot/data_pipelines/data.json'
```
### Step 4: Testing with local inference

```bash
python recipes/inference/local_inference/inference.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --peft_model chatbot-8b
```
