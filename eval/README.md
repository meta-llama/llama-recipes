# Llama Model Evaluation

Llama-Recipe make use of `lm-evaluation-harness` for evaluating our fine-tuned Llama2 model. It also can serve as a tool to evaluate quantized model to ensure the quality in lower precision or other optimization applied to the model that might need evaluation.


`lm-evaluation-harness` provide a wide range of [features](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#overview):

- Over 60 standard academic benchmarks for LLMs, with hundreds of subtasks and variants implemented.
- Support for models loaded via transformers (including quantization via AutoGPTQ), GPT-NeoX, and Megatron-DeepSpeed, with a flexible tokenization-agnostic interface.
- Support for fast and memory-efficient inference with vLLM.
- Support for commercial APIs including OpenAI, and TextSynth.
- Support for evaluation on adapters (e.g. LoRA) supported in HuggingFace's PEFT library.
- Support for local models and benchmarks.

The Language Model Evaluation Harness is also the backend for 🤗 [Hugging Face's (HF) popular Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). 

## Setup

Before running the evaluation script, ensure you have all the necessary dependencies installed.

### Dependencies

- Python 3.8+
- lm-evaluation-harness
- Your language model's dependencies

### Installation

Clone the lm-evaluation-harness repository and install it:

```bash
git clone https://github.com/matthoffner/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .

```

### Quick Test

To run evaluation for HuggingFace `Llama2 7B` model  on a single GPU please run the following,

```bash
python eval.py --model hf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf --tasks hellaswag --device cuda:0   --batch_size 8

```
Tasks can be extended by using `,` between them for example `--tasks hellaswag,arc`.

To set the number of shots you can use `--num_fewshot` to set the number for few shot evaluation. 

### PEFT Fine-tuned model Evaluation  

In case you have fine-tuned your model using PEFT you can set the PATH to the PEFT checkpoints using PEFT as part of model_args as shown below:

```bash
python eval.py --model hf --model_args pretrained=meta-llama/Llama-2-7b-hf,dtype="float",peft=../peft_output --tasks hellaswag --num_fewshot 10  --device cuda:0 --batch_size 8 
```

### Limit the number of examples in benchmarks

There has been an study from [IBM on efficient benchmarking of LLMs](https://arxiv.org/pdf/2308.11696.pdf), with main take a way that to identify if a model is performing poorly, benchmarking on wider range of tasks is more important than the number example in each task. This means you could run the evaluation harness with fewer number of example to have initial decision if the performance got worse from the base line. To limit the number of example here, it can be set using `--limit` flag with actual desired number. 

```bash
python eval.py --model hf --model_args pretrained=meta-llama/Llama-2-7b-hf,dtype="float",peft=../peft_output --tasks hellaswag --num_fewshot 10  --device cuda:0 --batch_size 8 --limit 100
```

### Reproducing HuggingFace Open-LLM-Leaderboard

Here, we provided a list of tasks from `Open-LLM-Leaderboard` which can be used by passing `--open-llm-leaderboard-tasks` instead of `tasks` to the `eval.py`. 

**NOTE** Make sure to run the bash script below to that will set the include paths in the config files. The script with prompt you to enter the path to the cloned lm-evaluation-harness repo.**You would need this step only for the first time**.

```bash

bash open_llm_eval_prep.sh

```
Now we can run the eval benchmark:

```bash
python eval.py --model hf --model_args pretrained=meta-llama/Llama-2-7b-hf,dtype="float",peft=../peft_output --num_fewshot 10  --device cuda:0 --batch_size 8 --limit 100 --open-llm-leaderboard-tasks
```

In the HF leaderboard, the [LLMs are evaluated on 7 benchmarks](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) from Language Model Evaluation Harness as described below:

- AI2 Reasoning Challenge (25-shot) - a set of grade-school science questions.
- HellaSwag (10-shot) - a test of commonsense inference, which is easy for humans (~95%) but challenging for SOTA models.
- MMLU (5-shot) - a test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more.
- TruthfulQA (0-shot) - a test to measure a model's propensity to reproduce falsehoods commonly found online. Note: TruthfulQA in the Harness is actually a minima a 6-shots task, as it is prepended by 6 examples systematically, even when launched using 0 for the number of few-shot examples.
- Winogrande (5-shot) - an adversarial and difficult Winograd benchmark at scale, for commonsense reasoning.
- GSM8k (5-shot) - diverse grade school math word problems to measure a model's ability to solve multi-step mathematical reasoning problems.
For all these evaluations, a higher score is a better score. We chose these benchmarks as they test a variety of reasoning and general knowledge across a wide variety of fields in 0-shot and few-shot settings.

### Customized Llama Model

In case you have customized the Llama model, for example a quantized version of model where it has different model loading from normal HF model, you can follow [this guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage) to add your model to the `eval.py` and run the eval benchmarks.

You can also find full task list [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).