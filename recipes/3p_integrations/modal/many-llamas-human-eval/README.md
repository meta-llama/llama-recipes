See `rune2e.sh` for info on how to run the experiment.

# Many Llamas Human Eval

In this directory, we run an experiment answering the question:

*If we run enough Llama models in parallel, can they outperform GPT-4o on HumanEval?*

It seeks to increase model performance not by scaling parameters, but by scaling compute time.

### Technical Blog

This experiment has been built and run by the team at [Modal](https://modal.com), and is described in the following blog post:

[Beat GPT-4o at Python by searching with 100 dumb LLaMAs](https://modal.com/blog/llama-human-eval)

The experiment has since been adapted to use the [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) model, and run end-to-end using the Modal serverless platform.

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

The resulting plots of the evals will be saved locally to:
- `/tmp/plot-pass-k.jpeg`
- `/tmp/plot-fail-k.jpeg`

`/tmp/plot-pass-k.jpeg` shows pass@k for the Llama 3.2 3B Instruct model vs pass@1 for GPT-4o. 

You'll see that at 100 generations, the Llama model is able to perform on-par with GPT-4o. At higher scale, the Llama model will outperform GPT-4o.

`/tmp/plot-fail-k.jpeg` shows fail@k across a log-scale, showing smooth scaling of this method.

