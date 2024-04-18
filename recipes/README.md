This folder contains examples organized by topic:

| Subfolder | Description |
|---|---|
[quickstart](./quickstart)|The "Hello World" of using Llama2, start here if you are new to using Llama2
[multilingual](./multilingual)|Scripts to add a new language to Llama2
[finetuning](./finetuning)|Scripts to finetune Llama2 on single-GPU and multi-GPU setups
[inference](./inference)|Scripts to deploy Llama2 for inference locally and using model servers
[use_cases](./use_cases)|Scripts showing common applications of Llama2
[responsible_ai](./responsible_ai)|Scripts to use PurpleLlama for safeguarding model outputs
[llama_api_providers](./llama_api_providers)|Scripts to run inference on Llama via hosted endpoints
[benchmarks](./benchmarks)|Scripts to benchmark Llama 2 models inference on various backends
[code_llama](./code_llama)|Scripts to run inference with the Code Llama models
[evaluation](./evaluation)|Scripts to evaluate fine-tuned Llama2 models using `lm-evaluation-harness` from `EleutherAI`


**<a id="replicate_note">Note on using Replicate</a>**
To run some of the demo apps here, you'll need to first sign in with Replicate with your github account, then create a free API token [here](https://replicate.com/account/api-tokens) that you can use for a while. After the free trial ends, you'll need to enter billing info to continue to use Llama2 hosted on Replicate - according to Replicate's [Run time and cost](https://replicate.com/meta/llama-2-13b-chat) for the Llama2-13b-chat model used in our demo apps, the model "costs $0.000725 per second. Predictions typically complete within 10 seconds." This means each call to the Llama2-13b-chat model costs less than $0.01 if the call completes within 10 seconds. If you want absolutely no costs, you can refer to the section "Running Llama2 locally on Mac" above or the "Running Llama2 in Google Colab" below.

**<a id="octoai_note">Note on using OctoAI</a>**
You can also use [OctoAI](https://octo.ai/) to run some of the Llama demos under [OctoAI_API_examples](./llama_api_providers/OctoAI_API_examples/). You can sign into OctoAI with your Google or GitHub account, which will give you $10 of free credits you can use for a month. Llama2 on OctoAI is priced at [$0.00086 per 1k tokens](https://octo.ai/pricing/) (a ~350-word LLM response), so $10 of free credits should go a very long way (about 10,000 LLM inferences).

### [Running Llama2 in Google Colab](https://colab.research.google.com/drive/1-uBXt4L-6HNS2D8Iny2DwUpVS4Ub7jnk?usp=sharing)
To run Llama2 in Google Colab using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), download the quantized Llama2-7b-chat model [here](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf), or follow the instructions above to build it, before uploading it to your Google drive. Note that on the free Colab T4 GPU, the call to Llama could take more than 20 minutes to return; running the notebook locally on M1 MBP takes about 20 seconds.
