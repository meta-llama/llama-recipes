## End to End Steps to create a Chatbot using Retrieval Augmented Fine Tuning(RAFT)

### Step 1 : Prepare related documents

Download all your desired docs in PDF, Text or Markdown format to "data" folder inside the data_pipelines folder.

In this case we have an example of [Getting started with Meta Llama](https://llama.meta.com/get-started/) and other llama related documents such Llama3, Purple Llama, Code Llama papers. Ideally, we should have searched all Llama documents across the web and follow the procedure below on them but that would be very costly for the purpose of a tutorial, so we will stick to our limited documents here. In this case, we want to use Llama FAQ as eval data so we should not put it into the data folder for training.

### Step 2 : Prepare RAFT dataset for fine-tuning

To use Meta Llama 3 70B model for the RAFT datasets creation from the prepared documents, we can either use Meta Llama 3 70B APIs from LLM cloud providers or host local LLM server.

In this example, we can use OctoAI API as a demo, and the APIs could be replaced by any other API from other providers.

**NOTE** The generated data by these APIs or the model needs to be vetted to make sure about the quality.

```bash
export OCTOAI_API_TOKEN="OCTOAI_API_TOKEN"
python generate_question_answers.py
```

**NOTE** You need to be aware of your RPM (requests per minute), TPM (tokens per minute) and TPD (tokens per day), limit on your account in case using any of model API providers. In our case we had to process each document at a time. Then merge all the Q&A `json` files to make our dataset. We aimed for a specific number of Q&A pairs per document anywhere between 50-100. This is experimental and totally depends on your documents, wealth of information in them and how you prefer to handle question, short or longer answers etc.

Alternatively we can use on prem solutions such as the [TGI](../../../../inference/model_servers/hf_text_generation_inference/README.md) or [VLLM](../../../../inference/model_servers/llama-on-prem.md). Here we will use the prompt in the [generation_config.yaml](./generation_config.yaml) to instruct the model on the expected format and rules for generating the Q&A pairs. In this example, we will show how to create a vllm openai compatible server that host Meta Llama 3 70B instruct locally, and generate the RAFT dataset.

```bash
# Make sure VLLM has been installed
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server  --model meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size 2 --disable-log-requests --port 8001
```

**NOTE** Please make sure the port has not been used. Since Meta Llama3 70B instruct model requires at least 135GB GPU memory, we need to use multiple GPUs to host it in a tensor parallel way.

Once the server is ready, we can query the server given the port number 8001 in another terminal. Here, "-v" sets the port number and "-t" sets the number of questions we ask the Meta Llama3 70B Instruct model to generate per chunk.

```bash
python raft.py -v 8001 -t 5
```

This python program will read all the documents inside of "data" folder and transform the text into embeddings and split the data into batches by the SemanticChunker. Then we apply the question_prompt_template, defined in "raft.yaml", to each batch, and finally we will use each batch to query VLLM server and save the return a list of question list for all batches.

We now have a related context as text chunk and a corresponding question list. For each question in the question list, we want to generate a Chain-of-Thought (COT) style question using Llama 3 70B Instruct as well. Once we have the COT answers, we can start to make a dataset that contains "instruction" which includes some unrelated chunks called distractor and has a probability P to include the related chunk.

```python
{
  'id': 'seed_task_0',
  'type': 'general',
  'question': 'What is the official motto of the United States of America?',
  'context': {
    'sentences': [
      ["the Gulf of Mexico are prone to hurricanes, ... and enforces the Act. [ 189 ] As of 2022, the U. S",
    "energy from fossil fuel and the largest ... there are 19, 969 airports in the U. S., of which 5, 193 are designated",
    'weaponry, ideology, and international i... and is a permanent member of the UN Security Council. The first documentary evidence of the phrase " United States',
    '[CLS] United States of America Flag Coat of arms ... dominance in nuclear and conventional',
    '##om ic soft pow er. [ 405 ] [ 406 ] Nearly all present ... rights in the United States are advanced by global standards.']
    ],
    'title': [
      ['placeholder_title',
      'placeholder_title',
      'placeholder_title',
      'placeholder_title',
      'placeholder_title']
    ]
  },
  'answer': '"In God We Trust"',
  'cot_answer': None
}


```
### Step 3: Run the fune-tuning
Once the dataset is ready, we can start the fine-tuning step using the following commands in the llama-recipe main folder:

For distributed fine-tuning:
```bash
CUDA_VISIBLE_DEVICES=0,1  torchrun --nnodes 1 --nproc_per_node 2  recipes/finetuning/finetuning.py --use_peft --enable_fsdp --peft_method lora  --model_name meta-llama/Meta-Llama-3-8B-Instruct --output_dir raft-8b --num_epochs 5 --batch_size_training 4 --dataset "custom_dataset" -custom_dataset.test_split "test" --custom_dataset.file "recipes/finetuning/datasets/raft_dataset.py" --use-wandb  --run_validation True  --custom_dataset.data_path 'recipes/use_cases/end2end-recipes/raft/raft.jsonl'
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
python -m vllm.entrypoints.openai.api_server  --model meta-llama/Meta-Llama-3-8B-Instruct --enable-lora --lora-modules raft-8b=./raft-8b --port 8000  --disable-log-requests
```

**NOTE** If encounter import error: "ImportError: punica LoRA kernels could not be imported.", this means that VLLM must be installed with punica LoRA kernels to support LoRA adapter, please use following commands to install the VLLM from source.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_INSTALL_PUNICA_KERNELS=1 pip install -e .
```

On another terminal, we can go to the recipes/use_cases/end2end-recipes/chatbot/pipelines folder to start our eval script.

```bash
python eval_raft.py -m raft-8b -v 8000
```



Lastly, we can use another Meta Llama 3 70B Instruct model as a judge to compare the answer from the fine-tuned 8B model with the groud truth and get a score. To do this, we need to host another Meta Llama 3 70B Instruct VLLM server locally with command, just make sure the port is not been used:

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server  --model meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size 2 --disable-log-requests --port 8002
```

Then we can pass the port to the eval script:

```bash
python eval_raft.py -m raft-8b -v 8000 -j 8002
```




### Step 5: Testing with local inference

Once we believe our fine-tuned model has passed our evaluation and we can deploy it locally to play with it by manually asking questions. We can do this by

```bash
python recipes/inference/local_inference/inference.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --peft_model chatbot-8b
```
