## End to End Steps to create a Chatbot using fine-tuning

### Step 1 : Prepare related documents

Download all your desired docs in PDF, Text or Markdown format to "data" folder inside the data_pipelines folder.

In this case we have an example of [Getting started with Meta Llama](https://llama.meta.com/get-started/) and other llama related documents such Llama3, Purple Llama, Code Llama papers. Ideally, we should have searched all Llama documents across the web and follow the procedure below on them but that would be very costly for the purpose of a tutorial, so we will stick to our limited documents here. In this case, we want to use Llama FAQ as eval data so we should not put it into the data folder for training.

TODO: Download conversations in the Llama github issues and use it as training data.
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
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server  --model meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size 2 --disable-log-requests --port 8001
```

**NOTE** Please make sure the port has not been used. Since Meta Llama3 70B instruct model requires at least 135GB GPU memory, we need to use multiple GPUs to host it in a tensor parallel way.

Once the server is ready, we can query the server given the port number 8001 in another terminal. Here, "-v" sets the port number and "-t" sets the total questions we ask the Meta Llama3 70B instruct model to initially generate, but the model can choice to generate less questions if it can not found any Llama related context to avoid the model generate questions that too trivial and unrelated.

```bash
python generate_question_answers.py -v 8001 -t 1000
```

This python program will read all the documents inside of "data" folder and split the data into batches by the context window limit (8K for Meta Llama3 and 4K for Llama 2) and apply the question_prompt_template, defined in "generation_config.yaml", to each batch. Then it will use each batch to query VLLM server and save the return QA pairs and the contexts. Additionally, we will add another step called self-curation (see more details in [Self-Alignment with Instruction Backtranslation](https://arxiv.org/abs/2308.06259)), which uses another 70B model to evaluate whether a QA pair is based on the context and provides relevant information about Llama language models given that context. We will then save all the QA pairs that passed the evaluation into data.json file as our final fine-tuning training set.

Example of QA pair that did not pass the self-curation, in this case the QA pair did not focus on Llama model:
```json
{'Question': 'What is the name of the pre-trained model for programming and natural languages?', 'Answer': 'CodeBERT', 'Context': 'Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou. CodeBERT: A pre-trained model for programming and natural languages. In EMNLP (Findings), volume EMNLP 2020 of Findings of ACL, pp. 15361547. Association for Computational Linguistics, 2020.'} {'Reason': 'The question and answer pair is not relevant to the context about Llama language models, as it discusses CodeBERT, which is not a Llama model.', 'Result': 'NO'}
```
### Step 3: Run the fune-tuning
In the llama-recipe main folder, we can start the fine-tuning step using the following commands:

For distributed fine-tuning:
```bash
CUDA_VISIBLE_DEVICES=0,1  torchrun --nnodes 1 --nproc_per_node 2  recipes/finetuning/finetuning.py --use_peft --enable_fsdp --peft_method lora  --model_name meta-llama/Meta-Llama-3-8B-Instruct --output_dir chatbot-8b --num_epochs 6 --batch_size_training 4 --dataset "custom_dataset" -custom_dataset.test_split "test" --custom_dataset.file "recipes/finetuning/datasets/chatbot_dataset.py" --use-wandb  --run_validation True  --custom_dataset.data_path 'recipes/use_cases/end2end-recipes/chatbot/pipelines/data.json'
```

For fine-tuning in single-GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python recipes/finetuning/finetuning.py --quantization --use_peft --peft_method lora  --model_name meta-llama/Meta-Llama-3-8B-Instruct --output_dir chatbot-8b --num_epochs 6 --batch_size_training 2 --dataset "custom_dataset" -custom_dataset.test_split "test" --custom_dataset.file "recipes/finetuning/datasets/chatbot_dataset.py" --use-wandb  --run_validation True  --custom_dataset.data_path 'recipes/use_cases/end2end-recipes/chatbot/pipelines/data.json'
```

For more details, please check the readme in the finetuning recipe.

### Step 4: Evaluating with local inference

Once we have the fine-tuned model, we now need to evaluate it to understand its performance. Normally, to create a evaluation set, we should first gather some questions and manually write the ground truth answer. In this case, we created a eval set based on the Llama [Troubleshooting & FAQ](https://llama.meta.com/faq/), where the answers are written by human experts. Then we pass the evalset question to our fine-tuned model to get the model generated answers. To compare the model generated answers with ground truth, we can use either traditional eval method, eg. calcucate rouge score, or use LLM to act like a judge to score the similarity of them.

First we need to start the VLLM servers to host our fine-tuned 8B model. Since we used peft library to get a LoRA adapter, we need to pass special arguments to VLLM to enable the LoRA feature. Now, the VLLM server actually will first load the original model, then apply our LoRA adapter weights.

```bash
python -m vllm.entrypoints.openai.api_server  --model meta-llama/Meta-Llama-3-8B-Instruct --enable-lora --lora-modules chatbot=./chatbot-8b --port 8000  --disable-log-requests
```

**NOTE** If encounter import error: "ImportError: punica LoRA kernels could not be imported.", this means that Vllm must be installed with punica LoRA kernels to support LoRA adapter, please use following commands to install the VLLM from source.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_INSTALL_PUNICA_KERNELS=1 pip install -e .
```

Then pass the eval_set json file into the VLLM servers and start the comparison evaluation. Notice that our model name is now called chatbot instead of meta-llama/Meta-Llama-3-8B-Instruct.

```bash
python eval_chatbot.py -m chatbot -v 8000
```
We can also quickly compare our fine-tuned chatbot model with original 8B model using

```bash
python eval_chatbot.py -m meta-llama/Meta-Llama-3-8B-Instruct -v 8000
```

TODO: evaluation using LLM as judge
### Step 5: Testing with local inference

Once we believe our fine-tuned model has passed our evaluation and we can deploy it locally to manually test it by manually asking questions.

```bash
python recipes/inference/local_inference/inference.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --peft_model chatbot-8b
```
