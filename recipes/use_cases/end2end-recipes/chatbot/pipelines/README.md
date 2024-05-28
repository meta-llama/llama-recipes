## End to End Steps to create a Chatbot using fine-tuning

### Step 1 : Prepare related documents

Download all your desired docs in PDF, Text or Markdown format to "data" folder inside the data_pipelines folder.

In this case we have an example of [Getting started with Meta Llama](https://llama.meta.com/get-started/) and other llama related documents such Llama3, Purple Llama, Code Llama papers. Ideally, we should have searched all Llama documents across the web and follow the procedure below on them but that would be very costly for the purpose of a tutorial, so we will stick to our limited documents here. In this case, we want to use Llama FAQ as eval data so we should not put it into the data folder for training.

### Step 2 : Prepare data (Q&A pairs) for fine-tuning

To use Meta Llama 3 70B model for the question and answer (Q&A) pair datasets creation from the prepared documents, we can either use Meta Llama 3 70B APIs from LLM cloud providers or host local LLM server.

In this example, we can use OctoAI API as a demo, and the APIs could be replaced by any other API from other providers.

**NOTE** The generated data by these APIs or the model needs to be vetted to make sure about the quality.

```bash
export OCTOAI_API_TOKEN="OCTOAI_API_TOKEN"
python generate_question_answers.py
```

**NOTE** You need to be aware of your RPM (requests per minute), TPM (tokens per minute) and TPD (tokens per day), limit on your account in case using any of model API providers. In our case we had to process each document at a time. Then merge all the Q&A `json` files to make our dataset. We aimed for a specific number of Q&A pairs per document anywhere between 50-100. This is experimental and totally depends on your documents, wealth of information in them and how you prefer to handle question, short or longer answers etc.

Alternatively we can use on prem solutions such as the [TGI](../../../../inference/model_servers/hf_text_generation_inference/README.md) or [VLLM](../../../../inference/model_servers/llama-on-prem.md). Here we will use the prompt in the [generation_config.yaml](./generation_config.yaml) to instruct the model on the expected format and rules for generating the Q&A pairs. In this example, we will show how to create a vllm openai compatible server that host Meta Llama 3 70B instruct locally, generate the Q&A pairs and apply self-curation to get the final dataset.

```bash
# Make sure VLLM has been installed
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server  --model meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size 2 --disable-log-requests --port 8001
```

**NOTE** Please make sure the port has not been used. Since Meta Llama3 70B instruct model requires at least 135GB GPU memory, we need to use multiple GPUs to host it in a tensor parallel way.

Once the server is ready, we can query the server given the port number 8001 in another terminal. Here, "-v" sets the port number and "-t" sets the total questions we ask the Meta Llama3 70B instruct model to initially generate, but the model can choice to generate less questions if it can not found any Llama related context to avoid the model generate questions that too trivial and unrelated.

```bash
python generate_question_answers.py -v 8001 -t 7000
```

This python program will read all the documents inside of "data" folder and split the data into batches by the context window limit (8K for Meta Llama3 and 4K for Llama 2) and apply the question_prompt_template, defined in "generation_config.yaml", to each batch. Then it will use each batch to query VLLM server and save the return QA pairs and the contexts.

Additionally, we will add another step called self-curation (see more details in [Self-Alignment with Instruction Backtranslation](https://arxiv.org/abs/2308.06259)), which uses another 70B model to evaluate whether a QA pair is based on the context and provides relevant information about Llama language models given that context. We will then save all the QA pairs that passed the evaluation into data.json file as our final fine-tuning training set.

Example of QA pair that did not pass the self-curation as the QA pair did not focus on Llama model:
```json
'Question': 'What is the purpose of the killall command in Linux?', 'Answer': 'To kill a process and all its child processes', 'Context': 'If you want to kill a process and all its child processes, you can use the killall command. For example: killall firefox'
'Reason': "The question and answer pair is not related to Llama language models, it's about a Linux command. The context provided is about Llama models, but the QA pair is about a Linux command, which is not relevant to the context.", 'Result': 'NO'
```

### Step 3: Run the fune-tuning
Once the dataset is ready, we can start the fine-tuning step using the following commands in the llama-recipe main folder:

For distributed fine-tuning:
```bash
CUDA_VISIBLE_DEVICES=0,1  torchrun --nnodes 1 --nproc_per_node 2  recipes/finetuning/finetuning.py --use_peft --enable_fsdp --peft_method lora  --model_name meta-llama/Meta-Llama-3-8B-Instruct --output_dir chatbot-8b --num_epochs 10 --batch_size_training 4 --dataset "custom_dataset" -custom_dataset.test_split "test" --custom_dataset.file "recipes/finetuning/datasets/chatbot_dataset.py" --use-wandb  --run_validation True  --custom_dataset.data_path 'recipes/use_cases/end2end-recipes/chatbot/pipelines/data.json'
```


For fine-tuning in single-GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python recipes/finetuning/finetuning.py --quantization --use_peft --peft_method lora  --model_name meta-llama/Meta-Llama-3-8B-Instruct --output_dir chatbot-8b --num_epochs 5 --batch_size_training 1 --dataset "custom_dataset" -custom_dataset.test_split "test" --custom_dataset.file "recipes/finetuning/datasets/chatbot_dataset.py" --use-wandb  --run_validation True  --custom_dataset.data_path 'recipes/use_cases/end2end-recipes/chatbot/pipelines/data.json'
```

If we want to continue the fine-tuning process after our evaluation step, we can use  --from_peft_checkpoint argument to resume the fine-tuning from PEFT checkpoint folder. For example, we can run:

```bash
CUDA_VISIBLE_DEVICES=0,1  torchrun --nnodes 1 --nproc_per_node 2  recipes/finetuning/finetuning.py --use_peft --enable_fsdp --from_peft_checkpoint chatbot-8b  --peft_method lora  --model_name meta-llama/Meta-Llama-3-8B-Instruct --output_dir chatbot-8b-continue --num_epochs 5 --batch_size_training 4 --dataset "custom_dataset" -custom_dataset.test_split "test" --custom_dataset.file "recipes/finetuning/datasets/chatbot_dataset.py" --use-wandb  --run_validation True  --custom_dataset.data_path 'recipes/use_cases/end2end-recipes/chatbot/pipelines/data.json'
```

For more details, please check the readme in the finetuning recipe.

### Step 4: Evaluating with local inference

Once we have the fine-tuned model, we now need to evaluate it to understand its performance. Normally, to create a evaluation set, we should first gather some questions and manually write the ground truth answer. In this case, we created a eval set mostly based on the Llama [Troubleshooting & FAQ](https://llama.meta.com/faq/), where the answers are written by human experts. Then we pass the evalset question to our fine-tuned model to get the model generated answers. To compare the model generated answers with ground truth, we can use either traditional eval method, eg. calcucate rouge score, or use LLM to act like a judge to score the similarity of them.

First we need to start the VLLM servers to host our fine-tuned 8B model. Since we used peft library to get a LoRA adapter, we need to pass special arguments to VLLM to enable the LoRA feature. Now, the VLLM server actually will first load the original model, then apply our LoRA adapter weights. Then we can feed the eval_set.json file into the VLLM servers and start the comparison evaluation. Notice that our finetuned model name is now called "chatbot" instead of "meta-llama/Meta-Llama-3-8B-Instruct".

```bash
python -m vllm.entrypoints.openai.api_server  --model meta-llama/Meta-Llama-3-8B-Instruct --enable-lora --lora-modules chatbot=./chatbot-8b --port 8000  --disable-log-requests
```

**NOTE** If encounter import error: "ImportError: punica LoRA kernels could not be imported.", this means that VLLM must be installed with punica LoRA kernels to support LoRA adapter, please use following commands to install the VLLM from source.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_INSTALL_PUNICA_KERNELS=1 pip install -e .
```

On another terminal, we can go to the recipes/use_cases/end2end-recipes/chatbot/pipelines folder to start our eval script.

```bash
python eval_chatbot.py -m chatbot -v 8000
```

We can also quickly compare our fine-tuned chatbot model with original Meta Llama 3 8B Instruct model using

```bash
python eval_chatbot.py -m meta-llama/Meta-Llama-3-8B-Instruct -v 8000
```


Lastly, we can use another Meta Llama 3 70B Instruct model as a judge to compare the answer from the fine-tuned 8B model with the groud truth and get a score. To do this, we need to host another Meta Llama 3 70B Instruct VLLM server locally with command, just make sure the port is not been used:

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server  --model meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size 2 --disable-log-requests --port 8001
```

Then we can pass the port to the eval script:

```bash
python eval_chatbot.py -m chatbot -v 8000 -j 8001
```

and similarily get the eval result for the original model:

```bash
python eval_chatbot.py -m meta-llama/Meta-Llama-3-8B-Instruct  -v 8000 -j 8001
```



### Step 5: Testing with local inference

Once we believe our fine-tuned model has passed our evaluation and we can deploy it locally to play with it by manually asking questions. We can do this by

```bash
python recipes/inference/local_inference/inference.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --peft_model chatbot-8b
```
