
# Prompt Migration

## Overview

The **Prompt Migration** toolkit helps you assess and adapt prompts across different language models, ensuring consistent performance and reliability. It includes benchmarking capabilities and evaluation tools to measure the effectiveness of prompt migrations.

## Project Structure

- `notebooks/`: Contains Jupyter notebooks for interactive prompt migration examples
  - `harness.ipynb`: Main notebook demonstrating the prompt migration workflow
- `benchmarks/`: Tools and scripts for performance evaluation
- `environment.yml`: Conda environment specification with all required dependencies

## Prerequisites

1. **Conda Environment**
   - [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed
   - Python 3.10
   - Create and activate the environment:
     ```bash
     conda env create -f environment.yml
     conda activate prompt-migration
     ```

2. **Setting Up vLLM for Inference**
   If you plan to use [vLLM](https://github.com/vllm-project/vllm) for model inference:
   ```bash
   pip install vllm
   ```
   To serve a large model (example: Meta’s Llama 3.3 70B Instruct), you might run:
   ```bash
   vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size=4
   ```
   Adjust the model name and `--tensor-parallel-size` according to your hardware and parallelization needs.

3. **Accessing Hugging Face Datasets**
   If you need to work with private or gated Hugging Face datasets, follow these steps:
   1. **Create a Hugging Face account (if you don’t have one):**
      Visit [Hugging Face](https://huggingface.co/) and create an account.
   2. **Authenticate via the Hugging Face CLI:**
      - Log in to Hugging Face:
        ```bash
        huggingface-cli login
        ```
      - Enter your Hugging Face credentials (username and token). You can generate or retrieve your token in your [Hugging Face settings](https://huggingface.co/settings/tokens).
   3. **Check Dataset Permissions:**
      Some datasets may require explicit permission from the dataset owner. If you continue to have access issues, visit the dataset page on Hugging Face to request or confirm your access rights.

## Key Dependencies

- **DSPy**: For prompt engineering and evaluation
- **LM-eval**: Evaluation framework for language models
- **PyTorch** and **Transformers**: For model inference

## Getting Started

1. **Activate your environment:**
   ```bash
   conda activate prompt-migration
   ```
2. **Start Jupyter notebook server:**
   ```bash
   jupyter notebook
   ```
3. **Open the main notebook:**
   Navigate to the `notebooks/harness.ipynb` in your browser to get started.
4. **Explore Benchmarks:**
   Use the scripts in the `benchmarks/` directory to evaluate your prompt migrations.

## License

This project is part of the **Llama Recipes** collection. Please refer to the main repository’s license for usage terms.
