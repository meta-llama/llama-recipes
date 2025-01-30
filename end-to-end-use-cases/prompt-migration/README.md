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

4. **Configure MMLU Benchmark:**
   In the notebook, modify the benchmark configuration to use MMLU:
   ```python
   from benchmarks import llama_mmlu # You can also choose other available from `benchmarks/`
   benchmark = llama_mmlu
   ```

5. **Run Optimization:**
   Choose an optimization level from the notebook and run the optimizer:
   ```python
   # Choose one: "light", "medium", or "heavy"
   optimizer = dspy.MIPROv2(metric=benchmark.metric, auto="medium")
   optimized_program = optimizer.compile(student, trainset=trainset)
   
   # View the optimized prompt and/or demos
   print("BEST PROMPT:\n", optimized_program.signature.instructions)
   print("BEST EXAMPLES:\n", optimized_program.predict.demos)
   ```

6. **Run base and optimized prompt on meta-evals:**
   Take the optimized prompt and examples and update your working directory:
   - Navigate to `llama-recipes/end-to-end-use-cases/benchmarks/llm_eval_harness/meta_eval/work_dir/mmlu/utils.py`
   - Open a new terminal and setup meta-evals environment following the readme in /meta_eval
   - Update the prompts list with your base and optimized prompts as the first two items
   ```python
   prompts = ["base_prompt", "optimized_prompt"] # Your base prompt and optimized prompt
   ```
   - run lm_eval twice, once for base prompt and once for optimized prompt by changing the `prompt` index in template as such:
   ```python
      template = f"<|start_header_id|>user<|end_header_id|>{prompt[0]}. Question: {question}\n {choice}\n<|eot_id|> \n\n<|start_header_id|>assistant<end_header_id|>"
   ```

7. **Explore Benchmarks:**
   Use the scripts in the `benchmarks/` directory to evaluate your prompt migrations.

## License

This project is part of the **Llama Recipes** collection. Please refer to the main repository’s license for usage terms.
