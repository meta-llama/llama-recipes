# Llama 2 On-Prem Inference Using vLLM and TGI

Enterprise customers may prefer to deploy Llama 2 on-prem and run Llama in their own servers. This tutorial shows how to use Llama 2 with [vLLM](https://github.com/vllm-project/vllm) and Hugging Face [TGI](https://github.com/huggingface/text-generation-inference), two leading open-source tools to deploy and serve LLMs, and how to create vLLM and TGI hosted Llama 2 instances with [LangChain](https://www.langchain.com/), an open-source LLM app development framework which we used for our earlier demo apps with Llama 2 running on [local Mac](https://github.com/facebookresearch/llama-recipes/blob/main/demo_apps/HelloLlamaLocal.ipynb) or [Replicate cloud](https://github.com/facebookresearch/llama-recipes/blob/main/demo_apps/HelloLlamaCloud.ipynb). See [here](https://medium.com/@rohit.k/tgi-vs-vllm-making-informed-choices-for-llm-deployment-37c56d7ff705) for a detailed comparison of vLLM and TGI.

We'll use the Amazon EC2 instance running Ubuntu with an A10G 24GB GPU as an example of running vLLM and TGI with Llama 2, and you can replace this with your own server to implement on-prem Llama 2 deployment.

The Colab notebook to connect via LangChain with Llama 2 hosted as the vLLM and TGI API services is [here](https://colab.research.google.com/drive/1rYWLdgTGIU1yCHmRpAOB2D-84fPzmOJg?usp=sharing), also shown in the sections below.

This tutorial assumes that you you have been granted access to the Meta Llama 2 on Hugging Face - you can open a Hugging Face Meta model page [here](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) to confirm that you see "Gated model You have been granted access to this model"; if you don't see the "granted access" message, simply follow the instructions under "Access Llama 2 on Hugging Face" in the page. 

You'll also need your Hugging Face access token which you can get at your Settings page [here](https://huggingface.co/settings/tokens).

## Setting up vLLM with Llama 2

On a terminal, run the following commands:

```
conda create -n vllm python=3.8
conda activate vllm
pip install vllm
```

Then run `huggingface-cli login` and copy and paste your Hugging Face access token to complete the login.

<!-- markdown-link-check-disable -->
There are two ways to deploy Llama 2 via vLLM, as a general API server or an OpenAI-compatible server (see [here](https://platform.openai.com/docs/api-reference/authentication) on how the OpenAI API authenticates, but you won't need to provide a real OpenAI API key when running Llama 2 via vLLM in the OpenAI-compatible mode).
<!-- markdown-link-check-enable -->

### Deploying Llama 2 as an API Server

Run the command below to deploy vLLM as a general Llama 2 service:

```
python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 5000 --model meta-llama/Llama-2-7b-chat-hf
```

Then on another terminal you can run:

```
curl http://localhost:5000/generate -d '{
        "prompt": "Who wrote the book Innovators dilemma?",
        "max_tokens": 300,
        "temperature": 0
    }'
```

to send a query (prompt) to Llama 2 via vLLM and get Llama's response:

> Who wrote the book Innovators dilemma? The book "Innovator's Dilemma" was written by Clayton M. Christensen. It was first published in 1997 and has since become a classic in the field of business and innovation. In the book, Christensen argues that successful companies often struggle to adapt to disruptive technologies and new market entrants, and that this struggle can lead to their downfall. He also introduces the concept of the "innovator's dilemma," which refers to the paradoxical situation in which a company's efforts to improve its existing products or services can actually lead to its own decline.

Now in your Llama client app, you can make an HTTP request as the `curl` command above to send a query to Llama and parse the response.

If you add the port 5000 to your EC2 instance's security group's inbound rules with the TCP protocol, then you can run this on your Mac/Windows for test:

```
curl http://<EC2_public_ip>:5000/generate -d '{
        "prompt": "Who wrote the book godfather?",
        "max_tokens": 300,
        "temperature": 0
    }'
```

Also, if you have multiple GPUs, you can add the `--tensor-parallel-size` argument when starting the server (see [here](https://vllm.readthedocs.io/en/latest/serving/distributed_serving.html) for more info). For example, the command below runs the Llama 2 13b-chat model on 4 GPUs:

```
python api_server.py --host 0.0.0.0 --port 5000 --model meta-llama/Llama-2-13b-chat-hf --tensor-parallel-size 4
```
With multiple GPUs, you can also run replica of models as long as your model size can fit into targeted GPU memory. For example, if you have two A10G with 24 GB memory, you can run two 7B Llama 2 models at the same time. This can be done by launching two api servers each targeting specific CUDA cores on different ports:
`CUDA_VISIBLE_DEVICES=0 python api_server.py --host 0.0.0.0 --port 5000  --model meta-llama/Llama-2-7b-chat-hf`
and
`CUDA_VISIBLE_DEVICES=1 python api_server.py --host 0.0.0.0 --port 5001  --model meta-llama/Llama-2-7b-chat-hf`
The benefit would be now you can balance incoming requests to both models, reaching higher batch size processing for a trade-off of generation latency.


### Deploying Llama 2 as OpenAI-Compatible Server

You can also deploy the vLLM hosted Llama 2 as an OpenAI-Compatible service to easily replace code using OpenAI API. First, run the command below:

```
python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 5000 --model meta-llama/Llama-2-7b-chat-hf
```

Then on another terminal, run:

```
curl http://localhost:5000/v1/completions -H "Content-Type: application/json" -d '{
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "prompt": "Who wrote the book Innovators dilemma?",
        "max_tokens": 300,
        "temperature": 0
    }'
```
and you'll see the following result:

> The book "Innovator's Dilemma" was written by Clayton M. Christensen. It was first published in 1997 and has since become a classic in the field of business and innovation. In the book, Christensen argues that successful companies often struggle to adapt to disruptive technologies and new market entrants, and that this struggle can lead to their downfall. He also introduces the concept of the "innovator's dilemma," which refers to the paradoxical situation in which a company's efforts to improve its existing products or services can actually lead to its own decline.

## Querying with Llama 2 via vLLM

On a Google Colab notebook, first install two packages:

```
!pip install langchain openai
```

Note that we only need to install the `openai` package with an `EMPTY` OpenAI API key to complete the LangChain integration with the OpenAI-compatible vLLM deployment of Llama 2. 

Then replace the <vllm_server_ip_address> below and run the code:

```
from langchain.llms import VLLMOpenAI

llm = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://<vllm_server_ip_address>:5000/v1",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    model_kwargs={
        "max_new_token": 300
        }
)

print(llm("Who wrote the book godfather?"))
```

You'll see an answer like:

> The book "The Godfather" was written by Mario Puzo. It was first published in 1969 and has since become a classic of American literature. The book was later adapted into a successful film directed by Francis Ford Coppola, which was released in 1972.

You can now use the Llama 2 instance `llm` created this way in any of the [Llama demo apps](https://github.com/facebookresearch/llama-recipes/tree/main/demo_apps) or your own Llama apps to integrate seamlessly with LangChain and LlamaIndex to build powerful on-prem Llama apps.

## Setting Up TGI with Llama 2

The easiest way to deploy Llama 2 with TGI is using its official docker image. First, replace `<your Hugging Face access token>` and set the three required shell variables (you may replace the `model` value above with another Llama 2 model):

```
model=meta-llama/Llama-2-13b-chat-hf
volume=$PWD/data
token=<your Hugging Face access token>
```

Then run the command below to deploy a quantized version of the Llama 2 13b-chat model with TGI:

```
docker run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.0.2 --model-id $model  --quantize bitsandbytes-nf4
```

After this, you'll be able to run the command below on another terminal:

```
curl 127.0.0.1:8080/generate -X POST -H 'Content-Type: application/json' -d '{
        "inputs": "Who wrote the book innovators dilemma?",
        "parameters": {
            "max_new_tokens":200
        }
    }'     
```

and see the answer generated by Llama 2 via TGI:

> The book "The Innovator's Dilemma" was written by Clayton Christensen, a professor at Harvard Business School. It was first published in 1997 and has since become a widely recognized and influential book on the topic of disruptive innovation.

## Querying with Llama 2 via TGI

Using LangChain to integrate with TGI-hosted Llama 2 is also straightforward. In the Colab above, first add a new code cell to install the Hugging Face `text_generation` package:

```
!pip install text_generation
```

Then add and run the code below:

```
llm = HuggingFaceTextGenInference(
    inference_server_url="http://<tgi_server_ip_address>:8080/",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)

llm("What wrote the book godfather?")
```

With the Llama 2 instance `llm` created this way, you can integrate seamlessly with LangChain and LlamaIndex to build powerful on-prem Llama 2 apps such as the [Llama demo apps](https://github.com/facebookresearch/llama-recipes/tree/main/demo_apps).

