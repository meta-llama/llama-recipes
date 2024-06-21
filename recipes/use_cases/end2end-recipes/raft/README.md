## Introduction:
As our Meta llama models become more popular, we noticed that there is a great demand to apply our Meta Llama models toward a custom domain to better serve the customers in that domain.
For example, a common scenario can be that a company has all the related documents in plain text for its custom domain and want to build chatbot that can help answer questions a client
could have.

Inspired by this demand,  we want to explore the possibilty of building a Github chatbot for llama-recipes based on Meta Llama models,
as a demo in this tutorial.  Even though our Meta Llama 3 70B Insturct model can be a great candidate, as it already has a excellent reasoning and knowledge, it is relatively costly to host in production.

Therefore, we want to explore the possibile ways to get a 8B-Instruct Meta Llama model based chatbot that can achieve the similar level of accuarcy of Meta Llama 70B-Instruct model based chatbot.
to save the inference cost.

## Understand the problems
To build a Github bot, we need to first understand what kind of questions that has been frequently asked. In our Github issues, we found out that the issues are not confined within Llama
model itself (eg, "where to download models"), but also include questions like quantization, training, inference problems which may related to Pytorch. Go through those questions can help us have a better understanding of what kind of data we need to collect.

Even though ideally we should included as many related documents as possible, such as Huggingface documentation, in this tutorial we will only include the Llama documents and Pytorch documents for demo purposes.

## Data Collections
Once we determine the domains we want to collect data from, we can start to think about what kind of data we want to collect and how to get that data. There are many llama related online conversation and disscusions in Reddit or Stack Overflow,
but the data cleaning will be hard, eg. filtering out unfaithful information.

In this tutorial, we want to use webpages in [Getting started with Meta Llama](https://llama.meta.com/get-started/)
along with webpages in [Pytorch blogs](https://pytorch.org/blog/) and [Pytorch tutorials](https://pytorch.org/tutorials/).

We can either use local folder or web crawl to get the data. For local folder option, we can download all the desired docs in PDF, Text or Markdown format to "data" folder.
Alternatively, we can create a sitemap xml, similar to the data_urls.xml example, and use Langchain SitemapLoader to get all the text in the webpages.

## Retrieval Augmented Fine Tuning (RAFT) concepts

In this tutorial, we want to use the a new method that combines finetuning with RAG called Retrieval Augmented Fine Tuning (RAFT).

RAFT is a general recipe to finetune a pretrained LLM to your domain-specific RAG settings.

## Create RAFT dataset
To use Meta Llama 3 70B model for the RAFT datasets creation from the prepared documents, we can either use Meta Llama 3 70B APIs from LLM cloud providers or host local LLM server.

We can use on prem solutions such as the [TGI](../../../../inference/model_servers/hf_text_generation_inference/README.md) or [VLLM](../../../../inference/model_servers/llama-on-prem.md). Here we will use the prompt in the [generation_config.yaml](./generation_config.yaml) to instruct the model on the expected format and rules for generating the Q&A pairs. In this example, we will show how to create a vllm openai compatible server that host Meta Llama 3 70B instruct locally, and generate the RAFT dataset.

```bash
# Make sure VLLM has been installed
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server  --model meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size 2 --disable-log-requests --port 8001
```

**NOTE** Please make sure the port has not been used. Since Meta Llama3 70B instruct model requires at least 135GB GPU memory, we need to use multiple GPUs to host it in a tensor parallel way.

Once the server is ready, we can query the server given the port number 8001 in another terminal. Here, "-u" sets the endpoint url to query and "-t" sets the number of questions we ask the Meta Llama3 70B Instruct model to generate per chunk. To use cloud API , please change the endpoint url to the cloud provider and set the api key using "-k". Here since we want to query our local hosted VLLM server, we can use following commend:

```bash
python raft.py -u "http://localhost:8001/v1" -k "EMPTY" -t 5
```

For cloud API key, we can also set it using system environment variables, such as

```bash
export API_KEY="THE_API_KEY_HERE"
python raft.py -u "CLOUD_API_URL" -t 5
```

**NOTE** When using cloud API, you need to be aware of your RPM (requests per minute), TPM (tokens per minute) and TPD (tokens per day), limit on your account in case using any of model API providers. This is experimental and totally depends on your documents, wealth of information in them and how you prefer to handle question, short or longer answers etc.

This python script will read all the documents either from local or web, and split the data into text chunks of 1000 charaters (defined by "chunk_size") using RecursiveCharacterTextSplitter.
Then we apply the question_prompt_template, defined in "raft.yaml", to each chunk, to get question list out of the text chunk.

We now have a related context as text chunk and a corresponding question list. For each question in the question list, we want to generate a Chain-of-Thought (COT) style question using Llama 3 70B Instruct as well.
Once we have the COT answers, we can start to make a dataset that where each sample contains "instruction" section includes some unrelated chunks called distractor and has a probability P to include the related chunk.

Here is a RAFT format json example from our saved raft.jsonl file. We have a "question" section for the generated question, "cot_answer" section for generated COT answers, where the final answer will be added after "<ANSWER>" token, and we also created a "instruction" section
that has all the documents included (each document splited by <DOCUMENT> <\/DOCUMENT> tag) and finally the question appended in the very end. This "instruction"
section will be the input during the training, and the "cot_answer" will be the output label that the loss will be calculated on.

```python
{
   "id":"seed_task_228",
   "type":"general",
   "question":"What is the context length supported by Llama 3 models?",
   "context":{
      "sentences":[
         [
            "DISTRACT_DOCS 1"
            "DISTRACT_DOCS 2"
            "We hope that Code Llama will inspire others to leverage Llama 2 to create new innovative tools for research and commercial products. Download the model Explore more on Code Llama Discover more about Code Llama here \u2014 visit our resources, ranging from our research paper, getting started guide and more. Code Llama GitHub repository Research paper Download the model Getting started guide Meta Llama 3 Build the future of AI with Meta Llama 3 Now available with both 8B and 70B pretrained and instruction-tuned versions to support a wide range of applications Build the future of AI with Meta Llama 3 Now available with both 8B and 70B pretrained and instruction-tuned versions to support a wide range of applications Get Started Experience Llama 3 on Meta AI Experience Llama 3 with Meta AI We\u2019ve integrated Llama 3 into Meta AI, our intelligent assistant, that expands the ways people can get things done, create and connect with Meta AI. You can see first-hand the performance of Llama 3 by using Meta AI for coding tasks and problem solving. Whether you're developing agents, or other AI-powered applications, Llama 3 in both 8B and 70B will offer the capabilities and flexibility you need to develop your ideas. Experience Llama 3 on Meta AI Enhanced performance Experience the state-of-the-art performance of Llama 3, an openly accessible model that excels at language nuances, contextual understanding, and complex tasks like translation and dialogue generation. With enhanced scalability and performance, Llama 3 can handle  multi-step tasks effortlessly, while our refined post-training processes significantly lower false refusal rates, improve response alignment, and boost diversity in model answers. Additionally, it drastically elevates capabilities like reasoning, code generation, and instruction following. Build the future of AI with Llama 3. Download Llama 3 Getting Started Guide With each Meta Llama request, you will receive: Meta Llama Guard 2 Getting started guide Responsible Use Guide Acceptable use policy Model card Community license agreement Benchmarks Llama 3 models take data and scale to new heights. It\u2019s been trained on our two recently announced custom-built 24K GPU clusters on over 15T token of data \u2013 a training dataset 7x larger than that used for Llama 2, including 4x more code. This results in the most capable Llama model yet, which supports a 8K context length that doubles the capacity of Llama 2. Model card Trust & safety A comprehensive approach to responsibility With the release of Llama 3, we\u2019ve updated the Responsible Use Guide (RUG) to provide the most comprehensive information on responsible development with LLMs. Our system-centric approach includes updates to our trust and safety tools with Llama Guard 2, optimized to support the newly announced taxonomy published by MLCommons expanding its coverage to a more comprehensive set of safety categories, Code Shield, and Cybersec Eval 2. In line with the principles outlined in our RUG , we recommend thorough checking and filtering of all inputs to and outputs from LLMs based on your unique content guidelines for your intended use case and audience. Meta Llama Guard 2 Explore more on Meta Llama 3 Introducing Meta Llama 3: The most capable openly available LLM to date Read the blog Meet Your New Assistant: Meta AI, Built With Llama 3 Learn more Meta Llama 3 repository View repository Model card Explore Meta Llama 3 License META LLAMA 3 COMMUNITY LICENSE AGREEMENT Meta Llama 3 Version Release Date: April 18, 2024 \u201c Agreement \u201d means the terms and conditions for use, reproduction, distribution and modification of the Llama Materials set forth herein. \u201c Documentation \u201d means the specifications, manuals and documentation accompanying Meta Llama 3 distributed by Meta at https:\/\/llama.meta.com\/get-started\/ .",
            "DISTRACT_DOCS 3"
            "DISTRACT_DOCS 4"
            "DISTRACT_DOCS 5"
         ]
      ],
      "title":[
         [
            "placeholder_title",
            "placeholder_title",
            "placeholder_title",
            "placeholder_title",
            "placeholder_title",
            "placeholder_title"
         ]
      ]
   },
   "oracle_context":"We hope that Code Llama will inspire others to leverage Llama 2 to create new innovative tools for research and commercial products. Download the model Explore more on Code Llama Discover more about Code Llama here \u2014 visit our resources, ranging from our research paper, getting started guide and more. Code Llama GitHub repository Research paper Download the model Getting started guide Meta Llama 3 Build the future of AI with Meta Llama 3 Now available with both 8B and 70B pretrained and instruction-tuned versions to support a wide range of applications Build the future of AI with Meta Llama 3 Now available with both 8B and 70B pretrained and instruction-tuned versions to support a wide range of applications Get Started Experience Llama 3 on Meta AI Experience Llama 3 with Meta AI We\u2019ve integrated Llama 3 into Meta AI, our intelligent assistant, that expands the ways people can get things done, create and connect with Meta AI. You can see first-hand the performance of Llama 3 by using Meta AI for coding tasks and problem solving. Whether you're developing agents, or other AI-powered applications, Llama 3 in both 8B and 70B will offer the capabilities and flexibility you need to develop your ideas. Experience Llama 3 on Meta AI Enhanced performance Experience the state-of-the-art performance of Llama 3, an openly accessible model that excels at language nuances, contextual understanding, and complex tasks like translation and dialogue generation. With enhanced scalability and performance, Llama 3 can handle  multi-step tasks effortlessly, while our refined post-training processes significantly lower false refusal rates, improve response alignment, and boost diversity in model answers. Additionally, it drastically elevates capabilities like reasoning, code generation, and instruction following. Build the future of AI with Llama 3. Download Llama 3 Getting Started Guide With each Meta Llama request, you will receive: Meta Llama Guard 2 Getting started guide Responsible Use Guide Acceptable use policy Model card Community license agreement Benchmarks Llama 3 models take data and scale to new heights. It\u2019s been trained on our two recently announced custom-built 24K GPU clusters on over 15T token of data \u2013 a training dataset 7x larger than that used for Llama 2, including 4x more code. This results in the most capable Llama model yet, which supports a 8K context length that doubles the capacity of Llama 2. Model card Trust & safety A comprehensive approach to responsibility With the release of Llama 3, we\u2019ve updated the Responsible Use Guide (RUG) to provide the most comprehensive information on responsible development with LLMs. Our system-centric approach includes updates to our trust and safety tools with Llama Guard 2, optimized to support the newly announced taxonomy published by MLCommons expanding its coverage to a more comprehensive set of safety categories, Code Shield, and Cybersec Eval 2. In line with the principles outlined in our RUG , we recommend thorough checking and filtering of all inputs to and outputs from LLMs based on your unique content guidelines for your intended use case and audience. Meta Llama Guard 2 Explore more on Meta Llama 3 Introducing Meta Llama 3: The most capable openly available LLM to date Read the blog Meet Your New Assistant: Meta AI, Built With Llama 3 Learn more Meta Llama 3 repository View repository Model card Explore Meta Llama 3 License META LLAMA 3 COMMUNITY LICENSE AGREEMENT Meta Llama 3 Version Release Date: April 18, 2024 \u201c Agreement \u201d means the terms and conditions for use, reproduction, distribution and modification of the Llama Materials set forth herein. \u201c Documentation \u201d means the specifications, manuals and documentation accompanying Meta Llama 3 distributed by Meta at https:\/\/llama.meta.com\/get-started\/ .",
   "cot_answer":"Here's the step-by-step reasoning to answer the question:\n\n1. The question asks about the context length supported by Llama 3 models.\n2. In the context, we need to find the relevant information about Llama 3 models and their context length.\n3. The relevant sentence is: \"This results in the most capable Llama model yet, which supports a 8K context length that doubles the capacity of Llama 2.\"\n##begin_quote## This results in the most capable Llama model yet, which supports a 8K context length that doubles the capacity of Llama 2. ##end_quote##\n4. From this sentence, we can see that Llama 3 models support a context length of 8K.\n\n<ANSWER>: 8K",
   "instruction":"<DOCUMENT> DISTRACT_DOCS 1 <\/DOCUMENT>...<DOCUMENT> DISTRACT_DOCS 5 <\/DOCUMENT>\nWhat is the context length supported by Llama 3 models?"
}
```
To create a evalset, ideally we should use human-annotation to create the question and answer pairs to make sure the the questions are related and answers are fully correct.
However, for demo purpose, we will use a subset of training json as the eval set. We can shuffle and random select 100 examples out of RAFT dataset. For evaluation purpose, we only need to keep the "question" section,
and the final answer section, marked by <ANSWER> tag in "cot_answer". Then we can manually check each example and remove those low-quaility examples, where the questions
are not related Llama or can not be infer without correct context. After the manual check, we keep 72 question and answer pairs as the eval_llama.json.

### Step 3: Run the fune-tuning
Once the RAFT dataset is ready in a json format, we can start the fine-tuning steps. Unfornately we found out that the LORA method did not produce a good result so we have to use the full fine-tuning using the following commands in the llama-recipe main folder:

```bash
torchrun --nnodes 1 --nproc_per_node 4  recipes/finetuning/finetuning.py --enable_fsdp --lr 1e-5 --context_length 8192 --num_epochs 1 --batch_size_training 1 --model_name meta-llama/Meta-Llama-3-8B-Instruct --dist_checkpoint_root_folder PATH_TO_ROOT_FOLDER --dist_checkpoint_folder fine-tuned  --use_fast_kernels --dataset "custom_dataset" --custom_dataset.test_split "test" --custom_dataset.file "recipes/finetuning/datasets/raft_dataset.py" --use-wandb  --run_validation True  --custom_dataset.data_path 'PATH_TO_RAFT_JSON'
```

Then convert the FSDP checkpoint to HuggingFace checkpoint using the following command:

```bash
python src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py --fsdp_checkpoint_path  PATH_TO_ROOT_FOLDER --consolidated_model_path PATH_TO_ROOT_FOLDER/fine-tuned-meta-llama --HF_model_path_or_name PATH_TO_ROOT_FOLDER

```

For more details, please check the readme in the finetuning recipe.

### Step 4: Evaluating with local inference

Once we have the fine-tuned model, we now need to evaluate it to understand its performance. We can use either traditional eval method, eg. calcucate exact match rate or rouge score.
In this tutorial, we can also use LLM to act like a judge to score model generated .


```bash
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server  --model raft-8b --port 8000  --disable-log-requests
```
**NOTE** If encounter import error: "ImportError: punica LoRA kernels could not be imported.", this means that VLLM must be installed with punica LoRA kernels to support LoRA adapter, please use following commands to install the VLLM from source.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_INSTALL_PUNICA_KERNELS=1 pip install -e .
```

On another terminal, we can use another Meta Llama 3 70B Instruct model as a judge to compare the answer from the fine-tuned 8B model with the groud truth and get a score. To do this, we need to host another Meta Llama 3 70B Instruct VLLM server locally with command, just make sure the port is not been used:

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server  --model meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size 2 --disable-log-requests --port 8002
```

Then we can pass the ports to the eval script:

```bash
CUDA_VISIBLE_DEVICES=1 python raft_eval.py -m raft-8b -v 8000 -j 8001 -r 5
```




### Step 5: Testing with local inference

Once we believe our fine-tuned model has passed our evaluation and we can deploy it locally to play with it by manually asking questions. We can do this by

```bash
python recipes/inference/local_inference/inference.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --peft_model chatbot-8b
```
