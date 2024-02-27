## Data Preprocessing Steps

### Step 1 : Prepare related documents

Download all your desired docs in PDF, Text or Markdown format to "data" folder

In this case we have an example of [Llama 2 Getting started guide](https://llama.meta.com/get-started/) and other llama related documents such Llama2, Purple Llama, Code Llama papers along with Llama FAQ. Ideally, we should have searched all Llama documents across the web and follow the procedure below on them but that would be very costly for the purpose of a tutorial, so we will stick to our limited documents here.

### Step 2 : Prepare data (Q&A pairs)

The idea here is to use OpenAI "gpt-3.5-turbo-16k" to create question and answer (Q&A) pair datasets from these documents. Here we will use the prompt in the [./config.yaml] to instruct the model on the expected format and rules for generating the Q&A pairs. This is only one way to handle this which is a popular method but beyond this any other preprocessing routine that help us making the Q&A pairs works. 

**Observation** during the dataset generation, we realized that "gpt-3.5-turbo-16k" tends to come up with abbreviations such as L-C stands for Llama fine-tuned for chat, that we had to literally ask not to include any abbreviations which still resulted in changing `L-C` to `Llama-C`. Adding "Never use any abbreviation." and "Do refer to Llama fine tuned chat model as Llama chat." still we got `Llama2-C` in the Q&A pairs. So instead of keep changing the prompts to see the impact ( its costly each time your generate new Q&A pairs) it was more cost effective to do some preprocessing and just to replace `Llama2-C` with `Llama chat` in the dataset.

**NOTE** The generated data by OpenAI model needs to be vetted to make sure about the quality.

```bash
export OPENAI_API_KEY="OPENAI_API_KEY"
python scrape_resources.py --url=https://llama.meta.com/get-started/
```

**NOTE** You need to be aware of your  RPM (requests per minute), TPM (tokens per minute) and TPD (tokens per day), limit on your OpenAI account. In our case we had to process each document at a time. Then merge all the Q&A `json` files to make our dataset. We aimed for a specific number of Q&A pairs per document anywhere between 50-100. This is experimental and totally depends on your documents, wealth of information in them and how you prefer to handle question, short or longer answers etc.

### Step 2 : Prepare dataset for fine-tuning Llama 2 Chat model

Here, as we want to fine-tune a chatbot model so its preferred to start with Llama 2 Chat model which already is instruction fine-tuned to serve as an assistant and further fine-tuned it for our Llama related data.


### Step 3: Run the training

```bash
torchrun --nnodes 1 --nproc_per_node 1  examples/finetuning.py  --use_peft --peft_method lora --quantization --model_name meta-llama/Llama-2-7b-chat-hf --output_dir ./peft-7b-quantized  --num_epochs 1 --batch_size 1 --dataset "custom_dataset" --custom_dataset.file "examples/llama_dataset.py"  --run_validation False  --custom_dataset.data_path '/data/home/hamidnazeri/llama-package/llama-recipes/tutorials/chatbot/data_pipelines/dataset.json'
```