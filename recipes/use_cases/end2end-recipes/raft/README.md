
## Introduction:
As our Meta Llama models become more popular, we noticed that there is a great demand to apply our Meta Llama models toward a custom domain to better serve the customers in that domain. For example, a common scenario can be that a company already has all the related documents in plain text for its custom domain and want to build chatbot that can help answer questions for its clients.

Inspired by this demand, we want to explore the possibility of building a Llama chatbot for our Llama users using Meta Llama models, as a demo in this tutorial.  Even though our Meta Llama 3 70B Instruct model can be a great candidate, as it already has a excellent reasoning and knowledge, it is relatively costly to host in production. Therefore, we want to produce a Meta Llama 8B Instruct model based chatbot that can achieve the similar level of accuracy of Meta Llama 70B-Instruct model based chatbot to save the inference cost.

## Data Collections
To build a Llama bot, we need to first collect the text data. Even though ideally we should included as many Llama related web documents as possible, in this tutorial we will only include the official documents for demo purposes. For example, we can use all the raw text from offical web pages listed in [Getting started with Meta Llama](https://llama.meta.com/get-started/) but we do not want to include our FAQ page as some of the eval questions will come from there.

We can either use local folder or web crawl to get the text data. For local folder option, we can download all the desired docs in PDF, Text or Markdown format to "data" folder, specified in the [raft.yaml](./raft.yaml).

Alternatively, we can create a sitemap xml, similar to the the following example, and use Langchain SitemapLoader to get all the text in the web pages.

```xml
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<loc>http://llama.meta.com/responsible-use-guide/</loc>
</url>
<url>
<loc>http://llama.meta.com/Llama2/</loc>
</url>
<url>
<loc>http://llama.meta.com/Llama2/license/</loc>
</url>
......
<url>
<loc>http://llama.meta.com/Llama2/use-policy/</loc>
</url>
<url>
<loc>http://llama.meta.com/code-Llama/</loc>
</url>
<url>
<loc>http://llama.meta.com/Llama3/</loc>
</url>
</urlset>
```
## Retrieval Augmented Fine Tuning (RAFT) concepts

In this tutorial, we want to introduce Retrieval Augmented Fine Tuning (RAFT) that combines finetuning with RAG to better utilize the custom domain text data.

RAFT is a general recipe to finetune a pretrained LLM to a domain-specific RAG settings. In RAFT, we prepare the training data such that each data point contains a question ( Q ), a set of documents (Dk), and a corresponding Chain-of-though style answer (A*) generated from one of the document (D*). We differentiate between two types of documents: oracle documents (D*) i.e. the documents from which the answer to the question can be deduced, and `distractor' documents (Di) that do not contain answer-relevant information, illustrated in the follwing graph:
![RAFT images](images/RAFT.png)

For more RAFT details, please check their [blog](https://gorilla.cs.berkeley.edu/blogs/9_raft.html)

## Create RAFT dataset

To use Meta Llama 3 70B model for the RAFT datasets creation from the prepared documents, we can either use Meta Llama 3 70B APIs from LLM cloud providers or host local LLM server.

We can use on prem solutions such as the [TGI](../../../../inference/model_servers/hf_text_generation_inference/README.md) or [VLLM](../../../../inference/model_servers/Llama-on-prem.md).

In this example, we will show how to create a vllm openai compatible server that host Meta Llama 3 70B instruct locally, and generate the RAFT dataset.

```bash
# Make sure VLLM has been installed
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server  --model meta-Llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size 2 --disable-log-requests --port 8001
```

**NOTE** Please make sure the port has not been used. Since Meta Llama3 70B instruct model requires at least 135GB GPU memory, we need to use multiple GPUs to host it in a tensor parallel way.

Once the server is ready, we can query the server given the port number 8001 in another terminal. Here, "-u" sets the endpoint url to query and "-t" sets the number of questions we ask the Meta Llama3 70B Instruct model to generate per chunk. To use cloud API , please change the endpoint url to the cloud provider and set the api key using "-k". Here since we want to query our local hosted VLLM server, we can use following command:

```bash
python raft.py -u "http://localhost:8001/v1" -k "EMPTY" -t 4
```

For cloud API key, we can also set it using system environment variables, such as

```bash
export API_KEY="THE_API_KEY_HERE"
python raft.py -u "CLOUD_API_URL" -t 4
```

**NOTE** When using cloud API, you need to be aware of your RPM (requests per minute), TPM (tokens per minute) and TPD (tokens per day), limit on your account in case using any of model API providers. This is experimental and totally depends on your documents, wealth of information in them and how you prefer to handle question, short or longer answers etc.

This [raft.py](./raft.py) will read all the documents either from local or web depending on the settings, and split the data into text chunks of 1000 characters (defined by "chunk_size") using RecursiveCharacterTextSplitter.

Then we apply the question_prompt_template, defined in [raft.yaml](./raft.yaml), to each chunk, to get question list out of the text chunk.

We now have a related context as text chunk and a corresponding question list. For each question in the question list, we want to generate a Chain-of-Thought (COT) style answer using Meta Llama 3 70B Instruct as well.

Once we have the COT answers, we can start to make a dataset where each sample contains "instruction" section that includes some unrelated chunks called distractor (by default we add 4 distractors). In the original RAFT method, there is a oracle probility P (by default 80%) that a related document will be included. This means that there is 1-P (by defualt 20%) chances that no related documents are provided, and the RAFT model should still try to predict COT_answer label, as the blog stated that "By removing the oracle documents in some instances of the training data, we are compelling the model to memorize domain-knowledge.".

In this tutorial we made a important modification by adding some additional refusal examples (by default this refusal probability is 5%) that when the related documents are not presented, we make the COT_answer label to be "Sorry, I don't know the answer to this question because related documents are not found. Please try again.". Our hyposis is that this will increase answer precision and reduce chatbot hallucination.  In real world production scenario, we prefer that the chatbot refuse to answer when no enough context are provided, so that we can detect this refusal signal and mitigate the risk of producing wrong or misleading answer, eg. we can ask for human agent to take over the conversation to better serve customers.

Here is a RAFT format json example from our saved raft.jsonl file. We have a "question" section for the generated question, "cot_answer" section for generated COT answers, where the final answer will be added after "<ANSWER>" token, and we also created a "instruction" section
that has all the documents included (each document splitted by <DOCUMENT> <\/DOCUMENT> tag) and finally the generated question appended in the very end. This "instruction" section will be the input during the fine-tuning, and the "cot_answer" will be the output label that the loss will be calculated on.

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
         ]
      ],
      "title":[
         [
            "placeholder_title",
            "placeholder_title",
            "placeholder_title",
            "placeholder_title",
            "placeholder_title",
         ]
      ]
   },
   "oracle_context":"We hope that Code Llama will inspire others to leverage Llama 2 to create new innovative tools for research and commercial products. Download the model Explore more on Code Llama Discover more about Code Llama here \u2014 visit our resources, ranging from our research paper, getting started guide and more. Code Llama GitHub repository Research paper Download the model Getting started guide Meta Llama 3 Build the future of AI with Meta Llama 3 Now available with both 8B and 70B pretrained and instruction-tuned versions to support a wide range of applications Build the future of AI with Meta Llama 3 Now available with both 8B and 70B pretrained and instruction-tuned versions to support a wide range of applications Get Started Experience Llama 3 on Meta AI Experience Llama 3 with Meta AI We\u2019ve integrated Llama 3 into Meta AI, our intelligent assistant, that expands the ways people can get things done, create and connect with Meta AI. You can see first-hand the performance of Llama 3 by using Meta AI for coding tasks and problem solving. Whether you're developing agents, or other AI-powered applications, Llama 3 in both 8B and 70B will offer the capabilities and flexibility you need to develop your ideas. Experience Llama 3 on Meta AI Enhanced performance Experience the state-of-the-art performance of Llama 3, an openly accessible model that excels at language nuances, contextual understanding, and complex tasks like translation and dialogue generation. With enhanced scalability and performance, Llama 3 can handle  multi-step tasks effortlessly, while our refined post-training processes significantly lower false refusal rates, improve response alignment, and boost diversity in model answers. Additionally, it drastically elevates capabilities like reasoning, code generation, and instruction following. Build the future of AI with Llama 3. Download Llama 3 Getting Started Guide With each Meta Llama request, you will receive: Meta Llama Guard 2 Getting started guide Responsible Use Guide Acceptable use policy Model card Community license agreement Benchmarks Llama 3 models take data and scale to new heights. It\u2019s been trained on our two recently announced custom-built 24K GPU clusters on over 15T token of data \u2013 a training dataset 7x larger than that used for Llama 2, including 4x more code. This results in the most capable Llama model yet, which supports a 8K context length that doubles the capacity of Llama 2. Model card Trust & safety A comprehensive approach to responsibility With the release of Llama 3, we\u2019ve updated the Responsible Use Guide (RUG) to provide the most comprehensive information on responsible development with LLMs. Our system-centric approach includes updates to our trust and safety tools with Llama Guard 2, optimized to support the newly announced taxonomy published by MLCommons expanding its coverage to a more comprehensive set of safety categories, Code Shield, and Cybersec Eval 2. In line with the principles outlined in our RUG , we recommend thorough checking and filtering of all inputs to and outputs from LLMs based on your unique content guidelines for your intended use case and audience. Meta Llama Guard 2 Explore more on Meta Llama 3 Introducing Meta Llama 3: The most capable openly available LLM to date Read the blog Meet Your New Assistant: Meta AI, Built With Llama 3 Learn more Meta Llama 3 repository View repository Model card Explore Meta Llama 3 License META LLAMA 3 COMMUNITY LICENSE AGREEMENT Meta Llama 3 Version Release Date: April 18, 2024 \u201c Agreement \u201d means the terms and conditions for use, reproduction, distribution and modification of the Llama Materials set forth herein. \u201c Documentation \u201d means the specifications, manuals and documentation accompanying Meta Llama 3 distributed by Meta at https:\/\/llama.meta.com\/get-started\/ .",
   "cot_answer":"Here's the step-by-step reasoning to answer the question:\n\n1. The question asks about the context length supported by Llama 3 models.\n2. In the context, we need to find the relevant information about Llama 3 models and their context length.\n3. The relevant sentence is: \"This results in the most capable Llama model yet, which supports a 8K context length that doubles the capacity of Llama 2.\"\n##begin_quote## This results in the most capable Llama model yet, which supports a 8K context length that doubles the capacity of Llama 2. ##end_quote##\n4. From this sentence, we can see that Llama 3 models support a context length of 8K.\n\n<ANSWER>: 8K",
   "instruction":"<DOCUMENT> DISTRACT_DOCS 1 <\/DOCUMENT>...<DOCUMENT> DISTRACT_DOCS 4 <\/DOCUMENT>\nWhat is the context length supported by Llama 3 models?"
}
```
To create a eval set, ideally we should use human-annotation to create the question and answer pairs to make sure the the questions are related and answers are fully correct.

However, this humman-annotation is costly and time-consuming. For demo purpose, we will use a subset of training json and our FAQ web page as the eval set. We can shuffle and random select 100 examples out of Llama RAFT dataset. For evaluation purpose, we only need to keep the "question" section, and the final answer section, marked by <ANSWER> tag in "cot_answer".

Then we can manually check each example and only pick the good examples. We want to make sure the questions are general enough that can be used to query the web search engine and are related Llama. Moreover, we also used some QA pairs, with some modification, from our FAQ page. Together, we created 72 question and answer pairs as the the eval set called eval_llama.json.

## Fune-tuning steps

Once the RAFT dataset is ready in a json format, we can start the fine-tuning steps. Unfortunately we found out that the LORA method did not produce a good result so we have to use the full fine-tuning method. We can use the following commands as an example in the Llama-recipes main folder:

```bash
export PATH_TO_ROOT_FOLDER = ./raft-8b
export PATH_TO_RAFT_JSON = recipes/use_cases/end2end-recipes/raft/output/raft.jsonl
torchrun --nnodes 1 --nproc_per_node 4  recipes/finetuning/finetuning.py --enable_fsdp --lr 1e-5 --context_length 8192 --num_epochs 1 --batch_size_training 1 --model_name meta-Llama/Meta-Llama-3-8B-Instruct --dist_checkpoint_root_folder $PATH_TO_ROOT_FOLDER --dist_checkpoint_folder fine-tuned  --use_fast_kernels --dataset "custom_dataset" --custom_dataset.test_split "test" --custom_dataset.file "recipes/finetuning/datasets/raft_dataset.py" --use-wandb  --run_validation True  --custom_dataset.data_path $PATH_TO_RAFT_JSON
```

For more details about multi-GPU finetuning, please check the [multigpu_finetuning.md](../../../finetuning/multigpu_finetuning.md) in the finetuning recipe.

Then we need to convert the FSDP checkpoint to HuggingFace checkpoint using the following command:

```bash
python src/Llama_recipes/inference/checkpoint_converter_fsdp_hf.py --fsdp_checkpoint_path  "$PATH_TO_ROOT_FOLDER/fine-tuned-meta-Llama/Meta-Llama-3-8B-Instruct" --consolidated_model_path "$PATH_TO_ROOT_FOLDER"
```

For more details about FSDP to HuggingFace checkpoint conversion, please check the [readme](../../../inference/local_inference/README.md) in the inference/local_inference recipe.

## Evaluation steps

Once we have the RAFT model, we now need to evaluate it to understand its performance. In this tutorial, we not only use traditional eval method, eg. calculate exact match rate or rouge score but also use LLM to act like a judge to score model generated.

We need to launch a VLLM server to host our converted model from PATH_TO_ROOT_FOLDER. To make things easier, we can rename the model folder raft-8b.
```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server  --model raft-8b --port 8000  --disable-log-requests
```

Similarly if we want to get 8B instruct baseline, we can launch a 8B model VLLM server instead:

```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server  --model  meta-Llama/Meta-Llama-3-8B-Instruct --port 8000  --disable-log-requests
```

On another terminal, we can use another Meta Llama 3 70B Instruct model as a judge to compare the answer from the RAFT 8B model with the ground truth and get a score. To do this, we need to host another Meta Llama 3 70B Instruct VLLM server locally with command, just make sure the port is not been used:

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server  --model meta-Llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size 2 --disable-log-requests --port 8001
```

Then we can pass the ports to the eval script to eval our raft model once our raft-8b vllm server is running:

```bash
CUDA_VISIBLE_DEVICES=4 python raft_eval.py -m raft-8b -u "http://localhost:8000/v1" -j "http://localhost:8001/v1" -r 5
```

To eval the 8B baseline we can use once our 8B vllm server is running:

```bash
CUDA_VISIBLE_DEVICES=4 python raft_eval.py -m meta-Llama/Meta-Llama-3-8B-Instruct -u "http://localhost:8000/v1" -j "http://localhost:8001/v1" -r 5
```

**NOTE** Please make sure the folder name in --model matches the "model_name" section in raft_eval_config.yaml. Otherwise VLLM will raise model not found error. By default, the RAFT model is called "raft-8b". Here "-u" specify the raft model endpoint url, "-j" specify the judge model endpoint url, "-r" defines how many top_k documents the RAG should retrieve.

This [raft_eval.py](./raft_eval.py) will load questions from eval set and generated answers from models and models+RAG. It will compare the generated answers with the ground truth to get the eval metrics, such as Rouge score or LLM_as_judge score, then save those metrics and eval details to logs.

## Experiment results

During our experiments, we did not get a good result from just using Llama website. We believe that our initial data from Llama website is not enough as it only has 327K characters and generates 1980+ RAFT examples. To increase our RAFT examples, we created another pytorch RAFT dataset with the text from offical web pages under [Pytorch blogs](https://pytorch.org/blog/) and [Pytorch tutorials](https://pytorch.org/tutorials/). This pytorch RAFT dataset has 20K RAFT examples generated from 4.7 million characters. Together, we have an all_data dataset that combines both Llama raft dataset and pytorch dataset. Then we fine-tuned the 8B model on those datasets separately for 1 epoch with learning rate of 1e-5 to get 3 RAFT models, namely Llama_only model, pytorch_only model and all_data model.  We used Llama website raw text as our RAG knowledge base and the document chunks_size is the same as the raft chunk_size 1000 characters.

We tested 5 models + RAG: all_data RAFT model, Llama_only RAFT model, pytorch_only RAFT model, 8B baseline, 70B baseline with the RAG document topk retrieve parameters of 3, 5 and 7. We used a Meta Llama 70B Instruct model as the judge to score our model generated answer with the ground truth in our eval set.

Here are the LLM_as_judge results:

![RAFT LLM_score comparison](images/LLM_score_comparison.png)

From the result, we noticed that RAFT models are performing very similarly to 8B baseline, noticeably worse than 70B baseline when context documents are limited (top_k <=5), but then RAFT models performs much better when top_k = 7, specially all_data 8B model already outperform 70B baseline (76.06% vs 74.65%).

Taking closer look at the number of refusal examples (when model saying “I do not know”). The all_data model is more cautious and tends to refuse to answer, where Llama_only_RAFT did not learn to refuse at all, because the Llama_only dataset only has 1980+ examples.

![Num of refusal comparison](images/Num_of_refusal_comparison.png)

We created a graph that shows the precision of our model answer, eg. when our RAFT model decides to answer, what is the likelihood of producing correct answers. Calculated by $\frac{LLMScore}{1-\frac{numRefusal}{totalQA}}$

Note that during our tests, the 8B and 70B baseline never refused to answer, so the precision of those models is the same as the LLM_score. We noticed that our RAFT models tend to refuse to answer when the provided documents are limited (top_k < 5), but if it decided to generate an answer, the likelyhood of being correct is higher. Specifically, when top_k =7, the all_data raft model has 82.97% likelihood of producing a correct answer when it decides to answer, far better than the 70B baseline of 74.65%.

![Answers Precision](images/Answers_Precision.png)

Here are some examples where our all_data RAFT can correctly answer while 70B failed:
```
Comparing interested question: What tokenizer is used as the basis for the special tokens in Meta Llama
ground_truth:  tiktoken
True all_data_RAG_answers: <ANSWER>: The tokenizer used as the basis for the special tokens in Meta Llama is tiktoken.
False 70B_RAG_answers: <ANSWER>: The tokenizer used as the basis for the special tokens in Meta Llama is SentencePiece.
```

```
Comparing interested question: What is the license under which the Llama Guard model and its weights are released?
groud_truth:  The license is the same as Llama 3, which can be found in the LICENSE file and is accompanied by the Acceptable Use Policy.
True raft-8b_RAG_answers: <ANSWER>: The license under which the Llama Guard model and its weights are released is the same as Llama 3, and the [LICENSE](../LICENSE) file contains more information about the license.
False 70B_RAG_answers: <ANSWER>: The Llama Guard model and its weights are licensed under the Llama 2 Community license.
```

Some learnings from these experiments:
1.Few thousands of RAFT examples did not yield a great result. From our experiments, above 10K RAFT examples is needed.
2.The LLM_as_judge is not always reliable, we noticed that some answers have been scored incorrectly.
3.The chunk_size for RAFT documents chunk and RAG document chunk should be the same.
4.RAFT method seems to help the LLM to differentiate the related documents from distractors rather than force the LLM to memorize the training data as we used Pytorch data as additional data to help our Llama chatbot to answer Llama questions. More research experiments will be needed to understand more about this.


## Local inference steps

Once we believe our RAFT model has passed our evaluation and we can deploy it locally to play with it by manually asking questions. We can do this by

```bash
python recipes/inference/local_inference/inference.py --model_name raft-8b
```

Lastly, special thanks to the first author of RAFT paper Tianjun Zhang to work together with us on this tutorial and provide many guidance during our experiments.
