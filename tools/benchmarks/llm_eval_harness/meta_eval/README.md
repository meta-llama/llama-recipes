
# Calculating Meta 3.1 Evaluation Metrics Using LM-Evaluation-Harness

As Llama models gain popularity, evaluating these models has become increasingly important. We have released all the evaluation details for Llama 3.1 models as datasets in the [3.1 evals Hugging Face collection](https://huggingface.co/collections/meta-llama/llama-31-evals-66a2c5a14c2093e58298ac7f). This recipe demonstrates how to calculate the Llama 3.1 reported benchmark numbers using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) library and our prompts from the 3.1 evals datasets on selected tasks.

## Disclaimer


1. **This recipe is not the official implementation** of Llama evaluation. Since our internal eval repo isn't public, we want to provide this recipe as an aid for anyone who wants to use the datasets we released. It is based on public third-party libraries, as this implementation is not mirroring Llama evaluation, therefore this may lead to minor differences in the produced numbers.
2. **Model Compatibility**: This tutorial is specifically for Llama 3 based models, as our prompts include Llama 3 special tokens, e.g. `<|start_header_id|>user<|end_header_id|>`. It will not work with models that are not based on Llama 3.

## Insights from Our Evaluation Process

Here are our insights about the differences in terms of the eval configurations and prompting methods between this implementation and Hugging Face [leaderboard implementation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/leaderboard).

- **Prompts**: We use Chain-of-Thought(COT) prompts while Hugging Face leaderboard does not. The prompts that define the output format are also different.
- **Metric calculation**: For MMLU-Pro, BBH, GPQA tasks, we ask the model to generate response and score the parsed answer from generated response, while Hugging Face leaderboard evaluation is comparing log likelihood of all label words, such as [ (A),(B),(C),(D) ].
- **Parsers**: For generative tasks, where the final answer needs to be parsed before scoring, the parser functions can be different between ours and Hugging Face leaderboard evaluation, as our prompts that define the model output format are designed differently.
- **Inference**: We use an internal LLM inference solution that does not apply padding, while Hugging Face leaderboard uses padding on the generative tasks (MATH and IFEVAL).
- **Tasks**  We run benchmarks on BBH and MMLU-Pro only for pretrained models and Math-Hard, IFeval, GPQA, only for pretrained models.

Given those differences, the numbers from this recipe can not be compared to the numbers in the Hugging Face [Open LLM Leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard), even if the task names are the same.

## Environment setups

Please install lm-evaluation-harness and our llama-recipe repo by following:

```
git clone git@github.com:meta-llama/llama-recipes.git
cd llama-recipes
pip install -U pip setuptools
pip install -e .
pip install lm-eval[math,ifeval,sentencepiece,vllm]==0.4.3
cd tools/benchmarks/llm_eval_harness/meta_eval
```

To access our [3.1 evals Hugging Face collection](https://huggingface.co/collections/meta-llama/llama-31-evals-66a2c5a14c2093e58298ac7f), you must:
- Log in to the Hugging Face website and click the 3.1 evals dataset pages and agree to the terms.
- Follow the [Hugging Face authentication instructions](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) to gain read access for your machine.

It is recommended to read the dataset card to understand the meaning of each column and use the viewer feature in the Hugging Face dataset to view our dataset. It is important to have some basic understanding of our dataset format and content before proceeding.

### Task Selection

Given the extensive number of tasks available (12 for pretrained models and 30 for instruct models), here we will focus on tasks that overlap with the popular Hugging Face [Open LLM Leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) as shown in the following:

- **Tasks for pretrained models**: BBH and MMLU-Pro
- **Tasks for instruct models**: Math-Hard, IFeval, GPQA, and MMLU-Pro

Here, we aim to get the benchmark numbers on the aforementioned tasks using Hugging Face [leaderboard implementation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/leaderboard). Please follow the instructions below to make necessary modifications to use our eval prompts and get more eval metrics.


### Run eval tasks

1. We created [eval_config.yaml](./eval_config.yaml) to store all the arguments and hyperparameters. This is the main config file you need to change if you want to eval other models, and a part of eval_config.yaml looks like this:

```yaml
model_name: "meta-llama/Llama-3.1-8B-Instruct" # The name of the model to evaluate. This must be a valid Llama 3 based model name in the HuggingFace model hub."

evals_dataset: "meta-llama/Llama-3.1-8B-Instruct-evals" # The name of the 3.1 evals dataset to evaluate, please make sure this eval dataset corresponds to the model loaded. This must be a valid Llama 3.1 evals dataset name in the Llama 3.1 Evals collection.
# Must be one of the following ["meta-llama/Llama-3.1-8B-Instruct-evals","meta-llama/Llama-3.1-70B-Instruct-evals","meta-llama/Llama-3.1-405B-Instruct-evals","meta-llama/Llama-3.1-8B-evals","meta-llama/Llama-3.1-70B-evals","meta-llama/Llama-3.1-405B-evals"]

tasks: "meta_instruct" # Available tasks for instruct model: "meta_math_hard", "meta_gpqa", "meta_mmlu_pro_instruct", "meta_ifeval"; or just use "meta_instruct" to run all of them.
# Available tasks for pretrain model: "meta_bbh", "meta_mmlu_pro_pretrain"; or just use "meta_pretrain" to run all of them.

tensor_parallel_size: 1 # The VLLM argument that speicify the tensor parallel size for the model, eg how many GPUs to use for a model copy.

data_parallel_size: 4 # The VLLM argument that speicify the data parallel size for the model, eg how copies of model will be used.

...

```

  Change `model_name` to the model name you want to eval on and change the `evals_dataset` according to the model type and parameters.  Remember to adjust the `tensor_parallel_size` to 2 or more to load the 70B models and change the `data_parallel_size` accordingly so that `tensor_parallel_size * data_parallel_size` is the number of GPUs you have. Please read the comments inside this yaml for detailed explanations on other parameters.

2. We already included all the related eval task yaml and python files in the [meta_template](./meta_template/) folder, which define all the task implementation. You do not need to change those manually, we will use [prepare_meta_eval.py](./prepare_meta_eval.py) to automatically change them later.

3. Then we can run [prepare_meta_eval.py](./prepare_meta_eval.py) that reads the configuration from [eval_config.yaml](./eval_config.yaml), copies everything in the template folder to a working folder `work_dir`, makes modification to those templates accordingly, prepares dataset if needed and prints out the CLI command to run the `lm_eval`.

  To run the [prepare_meta_eval.py](./prepare_meta_eval.py), we can do:

```
python prepare_meta_eval.py --config_path ./eval_config.yaml
```

  This script will load the default [eval_config.yaml](./eval_config.yaml) config and print out a `lm_eval` command to run `meta_instruct` group tasks,  which includes `meta_ifeval`, `meta_math_hard`, `meta_gpqa` and `meta_mmlu_pro_instruct`, for `meta-llama/Llama-3.1-8B-Instruct` model using `meta-llama/Llama-3.1-8B-Instruct-evals` dataset.

  An example output from [prepare_meta_eval.py](./prepare_meta_eval.py) looks like this:

```
lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=4,max_model_len=8192,add_bos_token=True,seed=42 --tasks meta_instruct --batch_size auto --output_path eval_results --include_path ./work_dir --seed 42  --log_samples
```

4. Then just copy the `lm_eval` command printed by [prepare_meta_eval.py](./prepare_meta_eval.py) back to your terminal and run it to get the result, which will be saved into `eval_results` folder by default.

**NOTE**: As for `--model vllm`, here we will use VLLM inference instead of Hugging Face inference because of the padding issue. By default, for the generative tasks, the `lm-eval --model_args="{...}" --batch_size=auto` command will use Hugging Face inference solution that uses a static batch method with [left padding](https://github.com/EleutherAI/lm-evaluation-harness/blob/8ad598dfd305ece8c6c05062044442d207279a97/lm_eval/models/huggingface.py#L773) using EOS_token for Llama models, but our internal evaluation will load python original checkpoints and handle individual generation request asynchronously without any padding. To simulate this, we will use VLLM inference solution to do dynamic batching without any padding.

**NOTE**: As for `add_bos_token=True`, since our prompts in the evals dataset has already included all the special tokens required by instruct model, such as `<|start_header_id|>user<|end_header_id|>`, we will not use `--apply_chat_template` argument for instruct models anymore. However, we need to use `add_bos_token=True` flag to add the BOS_token back during VLLM inference, as the BOS_token is removed by default in [this PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1465).

**NOTE**: For `meta_math_hard` tasks, some of our internal math ground truth has been converted to scientific notation, e.g. `6\sqrt{7}` has been converted to `1.59e+1`, which will be later handled by our internal math evaluation functions. As the lm-evaluation-harness [math evaluation utils.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/math/utils.py) can not fully handle those conversion, we will use the original ground truth from the original dataset [lighteval/MATH-Hard](https://huggingface.co/datasets/lighteval/MATH-Hard) by joining the tables on the original input questions. The `get_math_data` function in the [prepare_meta_eval.py](./prepare_meta_eval.py) will handle this step and produce a local parquet dataset file.

Moreover, we have modified this [math_hard/utils.py](./meta_template/math_hard/utils.py) to address two issues:

1. This python script only use [a regular expression "Final Answer: The final answer is(.*?). I hope it is correct."](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/math/utils.py#L192) to get the final answer, because this format is shown in the previous 4 shot examples prompts. However, our MATH Hard task is using 0 shot COT prompts that ask model to put the final answer into this string format `Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.` which can not be captured by previous regular expression, so we will use `\\box{}` to parse the final answer instead.

2. The [is_equiv(x1: str, x2: str)](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/math/utils.py#L144) function failed parse some ground truth, as we noticed some error logs like `[utils.py:158] couldn't parse one of [0,1) or [0,1)`, so all those questions will be marked as wrong. We raised [a issue to lm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/2212) about this problem and will add a string equality check statement before going to is_equiv() function for now as a temporary solution.


**NOTE**: For `meta_ifeval` tasks, we have to use the original configs, such as `instruction_id_list`, `kwargs`, from [wis-k/instruction-following-eval](https://huggingface.co/datasets/wis-k/instruction-following-eval) in order to use [lm-evaluation-harness IFeval evaluation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/leaderboard/ifeval). We will perform similar join back method using `get_ifeval_data` function in the [prepare_meta_eval.py](./prepare_meta_eval.py) to get a local parquet dataset file.


### Yaml Config Deep Dive

Here, we will use MMLU-Pro as a example to show the steps to create a yaml config with detailed explanations, so that people can follow this example to create other tasks configurations if they want. For more information, please read lm-evaluation-harness [new task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md).

**1.Define the config to load datasets**

We can use our 3.1 evals dataset as the source dataset and the corresponding subset and define the test split to latest. For example, if we want to calculate the MMLU_Pro metric for 3.1 8B instruct, the following configs are needed as explained below:

```yaml
task: meta_mmlu_pro_instruct
dataset_path: meta-llama/Llama-3.1-8B-Instruct-evals
dataset_name: Llama-3.1-8B-Instruct-evals__mmlu_pro__details
test_split: latest
```

If you want to run evaluation on 70B-Instruct, then it is recommended to change the `dataset_path` and  `dataset_name` from 8B to 70B, even though 70B-instruct and 8B-instruct share the same prompts, the `is_correct` column, which can be used to get the difference between current result and the reported results for each sample, is different.

**Note**: Config files for Llama-3.1-8B-Instruct are already provided in each task subfolder under [meta_template folder](./meta_template/). Remember to change the eval dataset name according to the model type and DO NOT use pretrained evals dataset on instruct models or vice versa.

**2.Configure preprocessing, prompts and ground truth**

Here is the example yaml snippet in the MMLU-Pro that handles dataset preprocess, prompts and ground truth.

```yaml
process_docs: !function utils.process_docs
doc_to_text: !function utils.doc_to_text
doc_to_target: gold
```

- `process_docs` : Defines the preprocess function for our datasets. In this case, we uses the `process_docs` python function that is defined in [utils.py](./meta_template/mmlu_pro/utils.py). This function will take the original dataset and output a processed dataset that has a out_doc, which contains `problem` which is the input question, `gold` which is the ground truth. We also renamed the `is_correct` column to `previously_is_correct` to allow detailed comparison for the difference of each sample between previously reported score and the current score. You must use eval dataset and model with same parameters and same model type to get a valid comparison.

-  `doc_to_text`: Defines the prompts. In the MMLU-Pro case, the `input_final_prompts` column always contains a list of a prompt, so we just use a python function that returns `input_final_prompts[0]`.

- `doc_to_target` Defines the ground truth, which in the MMLU-Pro case, is the `gold` that derived from `input_correct_responses[0]`.

**3.Configure task type and parser**

While Open LLM Leaderboard v2 uses [multiple choice format](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#multiple-choice-format) for MMLU-Pro, BBH, GPQA tasks by comparing log likelihood of all label words, such as [ (A),(B),(C),(D) ], we use generative task option, by asking the model to generate response in sentences given our carefully designed prompts, then using some parsers to grab the final answer, and scoring that final answer based on the ground truth. Here is a example config in the MMLU-Pro that enable the generative task and defines the regex parser:

```yaml
filter_list:
  - name: "strict-match"
    filter:
      - function: "regex"
        group_select: -1
        regex_pattern: 'best answer is ([A-Z])'
      - function: "take_first"
```
Since the MMLU-Pro task uses a 5-shot Chain-of-Thought(COT) prompts and the prompts are designed with explicitly instruction: "Your response should end with \"The best answer is [the_answer_letter].\" where the [the_answer_letter] is a letter from the provided choices.",  we will use a simple and intuitive regex expression `best answer is ([A-Z])` to parse the model response and take the last appearance as the final answer and this final answer will be scored based on the ground truth `gold` using exact match method.

**4.Define generation and metric config**

Then we need to define the generation and metric config, which looks like this:
```yaml
generation_kwargs:
  until: []
  do_sample: false
  temperature: 0
  max_gen_toks: 1024
num_fewshot: 0
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: true
```
Here we set the `num_fewshot` to 0 as our prompts have already been converted to 5-shots, and the model generation will only stop if the generated output tokens exceeds 1024, as stated in the [mmlu-pro eval details](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#mmlu-pro). We will set the `do_sample` to false and `temperature` to 0 as stated in our `eval_config` column in the dataset. We will use metric `exact_match` for this tasks and calculate the `mean` as our task aggregated number.

**NOTE**: While we tried our best to create the template files, those configs and functions are created based on public third-party library and are not exactly the same as our internal implementation, so there is a chance that the eval numbers are slightly different.

**NOTE**: We used the average of `inst_level_strict_acc,none` and `prompt_level_strict_acc,none` to get the final number for `IFeval` as stated [here](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about#task-evaluations-and-parameters).

**NOTE**: In the [Llama website](https://llama.com/), we reported the `macro_avg` metric, which is the average of all subtask average score, for `MMLU-Pro `task, but here we are calculating the `micro_avg` metric, which is the average score for all the individual samples, and those `micro_avg`  numbers can be found in the [eval_details.md](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#mmlu-pro).

**NOTE**: The eval numbers may be slightly different, as we observed around ±0.01 differences between each evaluation run because the latest VLLM inference is not very deterministic even with temperature=0. This behavior maybe related [this issue](https://github.com/vllm-project/vllm/issues/5404).
or it is expected due to 16-bits inference as stated in [this comment](https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535) and [this comment](https://github.com/vllm-project/vllm/issues/4112#issuecomment-2071115725).

## Acknowledgement

This tutorial is inspired by [leaderboard tasks implementation on the lm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/leaderboard) created by Hugging Face [Open LLM Leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) team.
We also extend our gratitude to the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) github repo from [EleutherAI](https://www.eleuther.ai/).
