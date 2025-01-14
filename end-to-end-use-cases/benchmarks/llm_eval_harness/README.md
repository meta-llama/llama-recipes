# Llama Model Evaluation

Llama-Recipe make use of `lm-evaluation-harness` for evaluating our fine-tuned Meta Llama3 (or Llama2) model. It also can serve as a tool to evaluate quantized model to ensure the quality in lower precision or other optimization applied to the model that might need evaluation.


`lm-evaluation-harness` provide a wide range of [features](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#overview):

- Over 60 standard academic benchmarks for LLMs, with hundreds of subtasks and variants implemented.
- Support for models loaded via [transformers](https://github.com/huggingface/transformers/) (including quantization via [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), and [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/), with a flexible tokenization-agnostic interface.
- Support for fast and memory-efficient inference with [vLLM](https://github.com/vllm-project/vllm).
- Support for commercial APIs including OpenAI and TextSynth.
- Support for evaluation on adapters (e.g. LoRA) supported in [HuggingFace's PEFT library](https://github.com/huggingface/peft).
- Support for local models and benchmarks.
- Evaluation with publicly available prompts ensures reproducibility and comparability between papers.
- Easy support for custom prompts and evaluation metrics.

The Language Model Evaluation Harness is also the backend for 🤗 [Hugging Face's (HF) popular Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

## Setup

Before running the evaluation, ensure you have all the necessary dependencies installed.

### Dependencies

- Python 3.8+
- Your language model's dependencies

### Installation

Clone the lm-evaluation-harness repository and install it:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

### Quick Test

To run evaluation for Hugging Face `Llama3.1 8B` model  on a single GPU please run the following,

```bash
lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B --tasks hellaswag --device cuda:0   --batch_size 8

```
Tasks can be extended by using `,` between them for example `--tasks hellaswag,arc`.

To set the number of shots you can use `--num_fewshot` to set the number for few shot evaluation.

### PEFT Fine-tuned model Evaluation

In case you have fine-tuned your model using PEFT you can set the PATH to the PEFT checkpoints using PEFT as part of model_args as shown below:

```bash
lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B, dtype="float",peft=../peft_output --tasks hellaswag --num_fewshot 10  --device cuda:0 --batch_size 8
```

### Limit the number of examples in benchmarks

There has been an study from [IBM on efficient benchmarking of LLMs](https://arxiv.org/pdf/2308.11696.pdf), with main take a way that to identify if a model is performing poorly, benchmarking on wider range of tasks is more important than the number example in each task. This means you could run the evaluation harness with fewer number of example to have initial decision if the performance got worse from the base line. To limit the number of example here, it can be set using `--limit` flag with actual desired number. But for the full assessment you would need to run the full evaluation. Please read more in the paper linked above.

```bash
lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B,dtype="float",peft=../peft_output --tasks hellaswag --num_fewshot 10  --device cuda:0 --batch_size 8 --limit 100
```

### Customized Llama Model

In case you have customized the Llama model, for example a quantized version of model where it has different model loading from normal HF model, you can follow [this guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage) to use `lm_eval.simple_evaluate()` to run the eval benchmarks.

You can also find full task list [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).

### Multi-GPU Evaluation with Hugging Face `accelerate`

`lm-evaluation-harness` support three main ways of using Hugging Face's [accelerate 🚀](https://github.com/huggingface/accelerate) library for multi-GPU evaluation.

To perform *data-parallel evaluation* (where each GPU loads a **separate full copy** of the model), `lm-evaluation-harness` leverage the `accelerate` launcher as follows:

```bash
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B \
    --tasks lambada_openai,arc_easy \
    --batch_size 16
```
(or via `accelerate launch --no-python lm_eval`).

For cases where your model can fit on a single GPU, this allows you to evaluate on K GPUs K times faster than on one.

**WARNING**: This setup does not work with FSDP model sharding, so in `accelerate config` FSDP must be disabled, or the NO_SHARD FSDP option must be used.

The second way of using `accelerate` for multi-GPU evaluation is when your model is *too large to fit on a single GPU.*

In this setting, run the library *outside the `accelerate` launcher*, but passing `parallelize=True` to `--model_args` as follows:

```
lm_eval --model hf \
    --tasks lambada_openai,arc_easy \
    --model_args pretrained=meta-llama/Llama-3.1-70B,parallelize=True \
    --batch_size 16
```

This means that your model's weights will be split across all available GPUs.

For more advanced users or even larger models, `lm-evaluation-harness` allows for the following arguments when `parallelize=True` as well:
- `device_map_option`: How to split model weights across available GPUs. defaults to "auto".
- `max_memory_per_gpu`: the max GPU memory to use per GPU in loading the model.
- `max_cpu_memory`: the max amount of CPU memory to use when offloading the model weights to RAM.
- `offload_folder`: a folder where model weights will be offloaded to disk if needed.

There is also an option to run with tensor parallel and data parallel together. This will allow you to take advantage of both data parallelism and model sharding, and is especially useful for models that are too large to fit on a single GPU.

```
accelerate launch --multi_gpu --num_processes {nb_of_copies_of_your_model} \
    -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-70B \
    --tasks lambada_openai,arc_easy \
    --model_args parallelize=True \
    --batch_size 16
```

To learn more about model parallelism and how to use it with the `accelerate` library, see the [accelerate documentation](https://huggingface.co/docs/transformers/v4.15.0/en/parallelism)

### Tensor + Data Parallel and Optimized Inference with `vLLM`

`lm-evaluation-harness` also support vLLM for faster inference on [supported model types](https://docs.vllm.ai/en/latest/models/supported_models.html), especially faster when splitting a model across multiple GPUs. For single-GPU or multi-GPU — tensor parallel, data parallel, or a combination of both — inference, for example:

```bash
lm_eval --model vllm \
    --model_args pretrained={model_name},tensor_parallel_size={GPUs_per_model},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size={model_replicas} \
    --tasks lambada_openai \
    --batch_size auto
```
To use vllm, do `pip install lm_eval[vllm]`. For a full list of supported vLLM configurations, please reference our [vLLM integration](https://github.com/EleutherAI/lm-evaluation-harness/blob/e74ec966556253fbe3d8ecba9de675c77c075bce/lm_eval/models/vllm_causallms.py) and the vLLM documentation.

vLLM occasionally differs in output from Huggingface. `lm-evaluation-harness` treat Huggingface as the reference implementation, and it provides a script for checking the validity of vllm results against HF.

> [!Tip]
> For fastest performance, `lm-evaluation-harness` recommend using `--batch_size auto` for vLLM whenever possible, to leverage its continuous batching functionality!

> [!Tip]
> Passing `max_model_len=4096` or some other reasonable default to vLLM through model args may cause speedups or prevent out-of-memory errors when trying to use auto batch size, such as for Mistral-7B-v0.1 which defaults to a maximum length of 32k.

For more details about `lm-evaluation-harness`, please visit checkout their github repo [README.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md).

## Calculating Meta 3.1 Evaluation Metrics Using LM-Evaluation-Harness

[meta_eval](./meta_eval/) folder provides a detailed guide on how to calculate the Meta Llama 3.1 evaluation metrics reported in our [Meta Llama website](https://llama.meta.com/) using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) and our [3.1 evals Huggingface collection](https://huggingface.co/collections/meta-llama/llama-31-evals-66a2c5a14c2093e58298ac7f). By following the steps outlined, users can replicate a evaluation process that is similar to Meta's, for specific tasks and compare their results with our reported metrics. While slight variations in results are expected due to differences in implementation and model behavior, we aim to provide a transparent method for evaluating Meta Llama 3 models using third party library. Please check the [README.md](./meta_eval/README.md) for more details.

## Reproducing HuggingFace Open-LLM-Leaderboard v2

In the HF leaderboard v2, the [LLMs are evaluated on 6 benchmarks](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) from Language Model Evaluation Harness as described below:

- **IFEval**: [IFEval](https://arxiv.org/abs/2311.07911) is a dataset designed to test a model’s ability to follow explicit instructions, such as “include keyword x” or “use format y.” The focus is on the model’s adherence to formatting instructions rather than the content generated, allowing for the use of strict and rigorous metrics.
- **BBH (Big Bench Hard)**: [BBH](https://arxiv.org/abs/2210.09261) is a subset of 23 challenging tasks from the BigBench dataset to evaluate language models. The tasks use objective metrics, are highly difficult, and have sufficient sample sizes for statistical significance. They include multistep arithmetic, algorithmic reasoning (e.g., boolean expressions, SVG shapes), language understanding (e.g., sarcasm detection, name disambiguation), and world knowledge. BBH performance correlates well with human preferences, providing valuable insights into model capabilities.
- **MATH**:  [MATH](https://arxiv.org/abs/2103.03874) is a compilation of high-school level competition problems gathered from several sources, formatted consistently using Latex for equations and asymptote for figures. Generations must fit a very specific output format. HuggingFace Open-LLM-Leaderboard v2 keeps only level 5 MATH questions and call it MATH Level 5.
- **GPQA (Graduate-Level Google-Proof Q&A Benchmark)**: [GPQA](https://arxiv.org/abs/2311.12022) is a highly challenging knowledge dataset with questions crafted by PhD-level domain experts in fields like biology, physics, and chemistry. These questions are designed to be difficult for laypersons but relatively easy for experts. The dataset has undergone multiple rounds of validation to ensure both difficulty and factual accuracy. Access to GPQA is restricted through gating mechanisms to minimize the risk of data contamination. Consequently, HuggingFace Open-LLM-Leaderboard v2 does not provide plain text examples from this dataset, as requested by the authors.
- **MuSR (Multistep Soft Reasoning)**: [MuSR](https://arxiv.org/abs/2310.16049) is a new dataset consisting of algorithmically generated complex problems, each around 1,000 words in length. The problems include murder mysteries, object placement questions, and team allocation optimizations. Solving these problems requires models to integrate reasoning with long-range context parsing. Few models achieve better than random performance on this dataset.
- **MMLU-PRO (Massive Multitask Language Understanding - Professional)**: [MMLU-Pro](https://arxiv.org/abs/2406.01574) is a refined version of the MMLU dataset, which has been a standard for multiple-choice knowledge assessment. Recent research identified issues with the original MMLU, such as noisy data (some unanswerable questions) and decreasing difficulty due to advances in model capabilities and increased data contamination. MMLU-Pro addresses these issues by presenting models with 10 choices instead of 4, requiring reasoning on more questions, and undergoing expert review to reduce noise. As a result, MMLU-Pro is of higher quality and currently more challenging than the original.

In order to install correct lm-evaluation-harness version, please check the Huggingface 🤗 Open LLM Leaderboard v2 [reproducibility section](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about#reproducibility).

To run a leaderboard evaluation for `Llama-3.1-8B`, we can run the following:

```bash
accelerate launch -m lm_eval --model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16  --log_samples --output_path eval_results --tasks leaderboard  --batch_size 4
```

Similarly to run a leaderboard evaluation for `Llama-3.1-8B-Instruct`, we can run the following, using `--apply_chat_template --fewshot_as_multiturn`:

```bash
accelerate launch -m lm_eval --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16  --log_samples --output_path eval_results --tasks leaderboard  --batch_size 4 --apply_chat_template --fewshot_as_multiturn
```

As for 70B models, it is required to run tensor parallelism as it can not fit into 1 GPU, therefore we can run the following for `Llama-3.1-70B-Instruct`:

```bash
lm_eval --model hf --batch_size 4 --model_args pretrained=meta-llama/Llama-3.1-70B-Instruct,parallelize=True --tasks leaderboard --log_samples --output_path eval_results --apply_chat_template --fewshot_as_multiturn
```
