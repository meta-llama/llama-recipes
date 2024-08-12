
# Reproducing Meta 3.1 Evaluation Metrics Using LM-Evaluation-Harness

As Meta Llama models gain popularity, evaluating these models has become increasingly important. We have released all the evaluation details for Meta-Llama 3.1 models as datasets in the [3.1 evals Huggingface collection](https://huggingface.co/collections/meta-llama/llama-31-evals-66a2c5a14c2093e58298ac7f). This tutorial demonstrates how to reproduce metrics similar to our reported numbers using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) library and our prompts from the 3.1 evals datasets on selected tasks.

## Important Notes

1. **This tutorial is not the official implementation** of Meta Llama evaluation. It is based on public third-party libraries, and the implementation may differ slightly from our internal evaluation, leading to minor differences in the reproduced numbers.
2. **Model Compatibility**: This tutorial is specifically for Llama 3 based models, as our prompts include Meta Llama 3 special tokens (e.g., `<|start_header_id|>user<|end_header_id|`). It will not work with models that are not based on Llama 3.


### Huggingface setups

In order to install correct lm-evaluation-harness version, please check the [Reproducibility section](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about#reproducibility) in Huggingface 🤗 Open LLM Leaderboard v2 About page. To add the VLLM dependency, we can do following:

```
git clone git@github.com:huggingface/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout adding_all_changess
pip install -e .[math,ifeval,sentencepiece,vllm]
```

To access our [3.1 evals Huggingface collection](https://huggingface.co/collections/meta-llama/llama-31-evals-66a2c5a14c2093e58298ac7f), you must:
- Log in to the Huggingface website and click the 3.1 evals dataset pages and agree to the terms.
- Follow the [Huggingface authentication instructions](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) to gain read access for your machine.

It is recommended to read the dataset card to understand the meaning of each column and use the viewer feature in the Huggingface dataset to view our dataset, such as this [MMLU-Pro](https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-8B-Instruct-evals/viewer/Meta-Llama-3.1-8B-Instruct-evals__mmlu_pro__details?row=0). It is important to have some basic understanding of our dataset format and content before proceeding.

### Task Selection

Given the extensive number of tasks available (12 for pretrained models and 30 for instruct models), this tutorial will focus on tasks that overlap with the popular Huggingface 🤗 [Open LLM Leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard), such as BBH and MMLU-Pro for pretrained models, and Math-Hard, IFeval, GPQA, and MMLU-Pro for instruct models. This tutorial serves as an example to demonstrate the reproduction process. The implementation will be based on the Huggingface 🤗 [leaderboard implementation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/leaderboard) and make necessary modifications to use our eval prompts and reproduce our reported metrics in [Meta Llama website](https://llama.meta.com/).


**NOTE**: There are many differences in terms of the eval configurations and prompts between this tutorial implementation and Huggingface 🤗 [leaderboard implementation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/leaderboard). For example, we use Chain-of-Thought(COT) prompts while Huggingface leaderboard does not, so the result numbers can not be apple to apple compared.

### Create task yaml

In order to use lm-evaluation-harness, we need to follow the lm-evaluation-harness [new task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md) to create a yaml file. We will use MMLU-Pro as a example to show the steps with detailed explanations:

**1.Define the config to load datasets**

We can use our 3.1 evals dataset as the source dataset and the corresponding subset and define the test split to latest. For example, if we want to reproduce the MMLU_Pro metric for 3.1 8B instruct, we should write the following yaml sections in the yaml:

```yaml
task: meta_mmlu_pro_instruct
dataset_path: meta-llama/Meta-Llama-3.1-8B-Instruct-evals
dataset_name: Meta-Llama-3.1-8B-Instruct-evals__mmlu_pro__details
test_split: latest
```


**Note**:Remember to change the eval dataset name according to the model type and DO NOT use pretrained evals dataset on instruct models or vice versa.

**2.Define the config for preprocessing, prompts and ground truth**

Here is the example yaml snippet in the MMLU-Pro that handles dataset preprocess, prompts and ground truth.

```yaml
process_docs: !function utils.process_docs
doc_to_text: !function utils.doc_to_text
doc_to_target: gold
```

- `process_docs` : Defines the preprocess function for our datasets. In this case, we uses the `process_docs` python function that is defined in [utils.py](./meta_template/mmlu_pro/utils.py). This function will take the original dataset and output a processed dataset that has a out_doc, which contains `problem` which is the input question, `gold` which is the ground truth. We also renamed the `is_correct` column to `previously_is_correct` to allow detailed comparison for the difference of each sample between previously reported score and the reproduced score. You must use eval dataset and model with same parameters and same model type to get a valid comparison.

-  `doc_to_text`: Defines the prompts. In the MMLU-Pro case, the `input_final_prompts` column always contains a list of a prompt, so we just use a python function that returns input_final_prompts[0].

- `doc_to_target` Defines the ground truth, which in the MMLU-Pro case, is the `gold` that derived from input_correct_responses[0].

**3.Define task type and parser**

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
Since the MMLU-Pro task uses a 5-shot Chain-of-Thought(COT) prompts and the prompts are designed with explicitly instruction of "Your response should end with \"The best answer is [the_answer_letter].\" where the [the_answer_letter] is a letter from the provided choices.",  we will use a simple and intuitive regex expression `best answer is ([A-Z])` to parse the model response and take the last appearance as the final answer and this final answer will be scored based on the ground truth `gold` using exact match method.

**Define generation and metric config**

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

In the end, we included all the yaml and python files in the [meta_template](./meta_template/) folder.

**NOTE**: While we tried our best to create the template files, those configs and functions are created based on public third-party library and are not exactly the same as our internal implementation, so there is a chance that the reproduced numbers are slightly different.

### Run eval tasks

Once we have the yaml created, we can run the tasks using `lm-eval` CLI and use the arguments defined in the [interface.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md). However, we have identified a major differences between our internal evaluation process and the recommended method `lm-eval --model_args="pretrained=<your_model>,revision=<your_model_revision>,dtype=<model_dtype>" --tasks=leaderboard  --batch_size=auto --output_path=<output_path>` by Open LLM Leaderboard in the [reproduce section](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about#reproducibility).

**Padding**

By default, for the generative tasks, the `lm-eval --model_args="{...}" --batch_size=auto` command will use Huggingface inference solution that uses a static batch method with [left padding](https://github.com/EleutherAI/lm-evaluation-harness/blob/8ad598dfd305ece8c6c05062044442d207279a97/lm_eval/models/huggingface.py#L773) using [EOS_token](https://huggingface.co/datasets/open-llm-leaderboard/meta-llama__Meta-Llama-3.1-8B-Instruct-details/blob/main/llhf__Meta-Llama-3.1-8B-Instruct/results_2024-07-19T10-53-21.449243.json#L2558) for Llama models. While our internal evaluation will load python original checkpoints and handle individual generation request asynchronously without any padding. To simulate this, we will use VLLM inference solution to do dynamic batching without any padding.

**NOTE**: Since our prompts in the evals dataset has already included all the special tokens required by instruct model, such as `|start_header_id|>user<|end_header_id|>`, we will not use `--apply_chat_template` argument for instruct models anymore. However, we need to use `add_bos_token=True` flag to add the BOS_token back during VLLM inference, as the BOS_token is removed by default in [this PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1465).

We create [eval_config.yaml](./eval_config.yaml) to store all the arguments and hyperparameters. Remember to adjust the `tensor_parallel_size` to 2 or more to load the 70B models and change the `data_parallel_size` accordingly so that `tensor_parallel_size X data_parallel_size` is the number of GPUs. Please read the comments inside this yaml for detailed explanations on other parameters. Then we can run a [meta_eval.py](meta_eval.py) that reads the configuration from [eval_config.yaml](./eval_config.yaml), copies everything in the template folder to a working folder `work_dir`, makes modification to those templates accordingly, prepares dataset if needed, run specified tasks and save the eval results to default `eval_results` folder.

To run the [meta_eval.py](meta_eval.py), we can do:

```
python meta_eval.py --config_path ./eval_config.yaml
```

This will load the default [eval_config.yaml](./eval_config.yaml) config and run a `meta_instruct` group tasks that includes `meta_ifeval`, `meta_math_hard`, `meta_gpqa` and `meta_mmlu_pro_instruct` tasks for `meta-llama/Meta-Llama-3.1-8B-Instruct` model using `meta-llama/Meta-Llama-3.1-8B-Instruct-evals` dataset.

**NOTE**: For `meta_math_hard` tasks, some of our internal math ground truth has been converted to scientific notation, eg. `6\sqrt{7}` has been converted to `1.59e+1` which will be later handled by our internal math evaluation functions. As the lm-evaluation-harness [math evaluation utils.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/math/utils.py) can not fully handle those conversion, we will use the original ground truth from the original dataset [lighteval/MATH-Hard](https://huggingface.co/datasets/lighteval/MATH-Hard) by joining the tables on the input questions. The `get_math_data` function in the [prepare_datasets.py](./prepare_dataset.py) will handle this step and produce a local parquet dataset file.

Moreover, we have modified this [math_hard/utils.py](./meta_template/math_hard/utils.py) to address two issues:

1. This python script only use [a regular expression "Final Answer: The final answer is(.*?). I hope it is correct."](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/math/utils.py#L192) to get the final answer, because this format is shown in the previous 4 shot examples prompts. However, our MATH Hard task is using 0 shot COT prompts that ask model to put the final answer into this string format `Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.` which can not be captured by previous regular expression, so we will use `\\box{}` to parse the final answer instead.

2. The [is_equiv(x1: str, x2: str)](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/math/utils.py#L144) function failed parse 78 ground truth, as we noticed some error logs like `[utils.py:158] couldn't parse one of [0,1) or [0,1)`, so all those questions will be marked as wrong. We will raise a issue about this problem and will add a string equality check statement before going to is_equiv() function for now as a temporal solution.


**NOTE**: For `meta_ifeval` tasks, we have to use the original configs, such as `instruction_id_list`, `kwargs`, from [wis-k/instruction-following-eval](https://huggingface.co/datasets/wis-k/instruction-following-eval) in order to use [lm-evaluation-harness IFeval evaluation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/leaderboard/ifeval). We will perform similar join back method using `get_ifeval_data` function in the [prepare_datasets.py](./prepare_dataset.py) to get a local parquet dataset file.

## Results

Here is the comparison between our reported numbers and the reproduced numbers in this tutorial:

| Model                        | MATH_HARD | GPQA_RAW | MMLU_PRO_RAW | IFeval  |
|------------------------------|-----------|----------|--------------|---------|
| 3.1 8B-Instruct reported     | 0.254     | 0.328    | 0.47         | 0.804   |
| 3.1 8B-Instruct reproduced   | 0.2417    | 0.3125   | 0.4675       | 0.7782  |
| 3.1 70B-Instruct reported    | 0.438     | 0.467    | 0.651        | 0.875   |
| 3.1 70B-Instruct reproduced  | 0.4388    | 0.4799   | 0.6475       | 0.848   |

| Model                  | BBH_RAW | MMLU_PRO_RAW |
|------------------------|---------|--------------|
| 3.1 8B reported        | 0.642   | 0.356        |
| 3.1 8B reproduced      | 0.6515  | 0.3572       |
| 3.1 70B reported       | 0.816   | 0.52         |
| 3.1 70B reproduced     | 0.8191  | 0.5225       |

From the table above, we can see that most of our reported results are very close to our reported number in the [Meta Llama website](https://llama.meta.com/).


**NOTE**: The reproduced numbers may be slightly different, as we observed around ±0.01 differences between each reproduce run because the latest VLLM inference is not very deterministic even with temperature=0. This behavior maybe related [this issue](https://github.com/vllm-project/vllm/issues/5404).
or it is expected as stated in [this comment](https://github.com/vllm-project/vllm/issues/4112#issuecomment-2071115725)

## Acknowledgement

This tutorial is inspired by [leaderboard tasks implementation on the lm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/leaderboard) created by Huggingface 🤗 [Open LLM Leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) team.
