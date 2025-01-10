# Many-Llamas Human-Eval

In this directory, we run an experiment answering the question:

*If we run enough Llama models in parallel, can they outperform GPT-4o on HumanEval?*

It seeks to increase model performance not through scaling parameters, but by scaling compute time.

### Technical Blog

This experiment built by the team at [Modal](https://modal.com), and is described in the following blog post:

[Beat GPT-4o at Python by searching with 100 small Llamas](https://modal.com/blog/llama-human-eval)

The experiment has since been upgraded to use the [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) model, and run end-to-end using the Modal serverless platform.

## Run it yourself

### Install the Modal CLI
From within your virtual environment, run:
```bash
pip install modal
```
And if you're new to Modal, authenticate with:
```bash
modal setup
# or if that doesn't work, try 
# python -m modal setup
```

That's all!

This CLI will execute your modal apps, which build and run containers on the cloud, on your GPU of choice.

### HuggingFace Pull Access

To download the model, you'll first need to accept the [Llama 3.2 License](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) on HuggingFace and be approved for access.

Then, create a [modal secret](https://modal.com/secrets) named `huggingface`, to which you'll add your HF_TOKEN as an environment variable.

### Run The Experiment

This command will run every step for you:
```bash
bash run_e2e.sh
```

Or if you prefer to run it manually, you can step through each of the modal commands in [the script](./run_e2e.sh).

This will execute:
1. Downloading the Llama 3.2 3B Instruct model to a cloud volume
2. Deploying a vLLM inference server to GPUs
3. Running hundreds of parallel generations on the HumanEval test set
4. Running the evaluation script to compute pass@k and fail@k
5. Generating graphs of pass@k and fail@k

### Results
<!-- markdown-link-check-disable -->
The resulting plots of the evals will be saved locally to:
- `/tmp/plot-pass-k.jpeg`
- `/tmp/plot-fail-k.jpeg`

`/tmp/plot-pass-k.jpeg` shows pass@k for the Llama 3.2 3B Instruct model vs pass@1 for GPT-4o. 

![plot-pass-k](https://github.com/user-attachments/assets/11e9dc6e-4322-4d44-b928-4ed7c4ce8262)

You'll see that at 100 generations, the Llama model is able to perform on-par with GPT-4o. At higher scale, the Llama model will outperform GPT-4o.

`/tmp/plot-fail-k.jpeg` shows fail@k across a log-scale, showing smooth scaling of this method.

![plot-fail-k](https://github.com/user-attachments/assets/7286e4ff-5090-4288-bd62-8a078c6dc5a1)
<!-- markdown-link-check-enable -->
