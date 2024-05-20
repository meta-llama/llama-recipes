# Benchmark Llama models on AWS

The [`FMBench`](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main) tool provides a quick and easy way to benchmark the Llama family of models for price and performance on any AWS service including [`Amazon SagMaker`](https://aws.amazon.com/solutions/guidance/generative-ai-deployments-using-amazon-sagemaker-jumpstart/), [`Amazon Bedrock`](https://aws.amazon.com/bedrock/) or `Amazon EKS` or `Amazon EC2` as `Bring your own endpoint`.

## The need for benchmarking

<!-- markdown-link-check-disable -->
Customers often wonder what is the best AWS service to run Llama models for _my specific use-case_ and _my specific price performance requirements_. While model evaluation metrics are available on several leaderboards ([`HELM`](https://crfm.stanford.edu/helm/lite/latest/#/leaderboard), [`LMSys`](https://chat.lmsys.org/?leaderboard)), but the price performance comparison can be notoriously hard to find and even more harder to trust. In such a scenario, we think it is best to be able to run performance benchmarking yourself on either on your own dataset or on a similar (task wise, prompt size wise) open-source datasets such as ([`LongBench`](https://huggingface.co/datasets/THUDM/LongBench), [`QMSum`](https://paperswithcode.com/dataset/qmsum)). This is the problem that [`FMBench`](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main) solves.
<!-- markdown-link-check-enable -->

## [`FMBench`](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main): an open-source Python package for FM benchmarking on AWS

`FMBench` runs inference requests against endpoints that are either deployed through `FMBench` itself (as in the case of SageMaker) or are available either as a fully-managed endpoint (as in the case of Bedrock) or as bring your own endpoint. The metrics such as inference latency, transactions per-minute, error rates and cost per transactions are captured and presented in the form of a Markdown report containing explanatory text, tables and figures. The figures and tables in the report provide insights into what might be the best serving stack (instance type, inference container and configuration parameters) for a given Llama model for a given use-case.

The following figure gives an example of the price performance numbers that include inference latency, transactions per-minute and concurrency level for running the `Llama2-13b` model on different instance types available on SageMaker using prompts for Q&A task created from the [`LongBench`](https://huggingface.co/datasets/THUDM/LongBench) dataset, these prompts are between 3000 to 3840 tokens in length. **_Note that the numbers are hidden in this figure but you would be able to see them when you run `FMBench` yourself_**.

![`Llama2-13b` on different instance types ](./img/business_summary.png)

The following table (also included in the report) provides information about the best available instance type for that experiment<sup>1</sup>.

|Information	|Value	|
|---	|---	|
|experiment_name	|llama2-13b-inf2.24xlarge	|
|payload_file	|payload_en_3000-3840.jsonl	|
|instance_type	|ml.inf2.24xlarge	|
|concurrency	|**	|
|error_rate	|**	|
|prompt_token_count_mean	|3394	|
|prompt_token_throughput	|2400	|
|completion_token_count_mean	|31	|
|completion_token_throughput	|15	|
|latency_mean	|**	|
|latency_p50	|**	|
|latency_p95	|**	|
|latency_p99	|**	|
|transactions_per_minute	|**	|
|price_per_txn	|**	|

<sup>1</sup> ** represent values hidden on purpose, these are available when you run the tool yourself.

The report also includes latency Vs prompt size charts for different concurrency levels. As expected, inference latency increases as prompt size increases but what is interesting to note is that the increase is much more at higher concurrency levels (and this behavior varies with instance types).

![Effect of prompt size on inference latency for different concurrency levels](./img/latency_vs_tokens.png)

### How to get started with `FMBench`

The following steps provide a [Quick start guide for `FMBench`](https://github.com/aws-samples/foundation-model-benchmarking-tool#quickstart). For a more detailed DIY version, please see the [`FMBench Readme`](https://github.com/aws-samples/foundation-model-benchmarking-tool?tab=readme-ov-file#the-diy-version-with-gory-details).

1. Each `FMBench` run works with a configuration file that contains the information about the model, the deployment steps, and the tests to run. A typical `FMBench` workflow involves either directly using an already provided config file from the [`configs`](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main/src/fmbench/configs) folder in the `FMBench` GitHub repo or editing an already provided config file as per your own requirements (say you want to try benchmarking on a different instance type, or a different inference container etc.).

    >A simple config file with key parameters annotated is included in this repo, see [`config.yml`](./config.yml). This file benchmarks performance of Llama2-7b on an `ml.g5.xlarge` instance and an `ml.g5.2xlarge` instance. You can use this provided config file as it is for this Quickstart.

1. Launch the AWS CloudFormation template included in this repository using one of the buttons from the table below. The CloudFormation template creates the following resources within your AWS account: Amazon S3 buckets, Amazon IAM role and an Amazon SageMaker Notebook with this repository cloned. A read S3 bucket is created which contains all the files (configuration files, datasets) required to run `FMBench` and a write S3 bucket is created which will hold the metrics and reports generated by `FMBench`. The CloudFormation stack takes about 5-minutes to create.

   |AWS Region                |     Link        |
   |:------------------------:|:-----------:|
   |us-east-1 (N. Virginia)    | [<img src="./img/CFT.png">](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=fmbench&templateURL=https://aws-blogs-artifacts-public.s3.amazonaws.com/artifacts/ML-FMBT/template.yml) |
   |us-west-2 (Oregon)    | [<img src="./img/CFT.png">](https://console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/new?stackName=fmbench&templateURL=https://aws-blogs-artifacts-public.s3.amazonaws.com/artifacts/ML-FMBT/template.yml) |

1. Once the CloudFormation stack is created, navigate to SageMaker Notebooks and open the `fmbench-notebook`.

1. On the `fmbench-notebook` open a Terminal and run the following commands.

    ```{.bash}
    conda create --name fmbench_python311 -y python=3.11 ipykernel
    source activate fmbench_python311;
    pip install -U fmbench
    ```

1. Now you are ready to `fmbench` with the following command line. We will use a sample config file placed in the S3 bucket by the CloudFormation stack for a quick first run.

    1. We benchmark performance for the `Llama2-7b` model on a `ml.g5.xlarge` and a `ml.g5.2xlarge` instance type, using the `huggingface-pytorch-tgi-inference` inference container. This test would take about 30 minutes to complete and cost about $0.20.

    1. It uses a simple relationship that 750 words equals 1000 tokens, to get a more accurate representation of token counts use the `Llama2 tokenizer`. **_It is strongly recommended that for more accurate results on token throughput you use a tokenizer specific to the model you are testing rather than the default tokenizer. See instructions provided [here](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main?tab=readme-ov-file#the-diy-version-with-gory-details) on how to use a custom tokenizer_**.

        <!-- markdown-link-check-disable -->
        ```{.bash}
        account=`aws sts get-caller-identity | jq .Account | tr -d '"'`
        region=`aws configure get region`
        fmbench --config-file s3://sagemaker-fmbench-read-${region}-${account}/configs/llama2/7b/config-llama2-7b-g5-quick.yml >> fmbench.log 2>&1
        ```
        <!-- markdown-link-check-enable -->

    1. Open another terminal window and do a `tail -f` on the `fmbench.log` file to see all the traces being generated at runtime.

        ```{.bash}
        tail -f fmbench.log
        ```

1. The generated reports and metrics are available in the `sagemaker-fmbench-write-<replace_w_your_aws_region>-<replace_w_your_aws_account_id>` bucket. The metrics and report files are also downloaded locally and in the `results` directory (created by `FMBench`) and the benchmarking report is available as a markdown file called `report.md` in the `results` directory. You can view the rendered Markdown report in the SageMaker notebook itself or download the metrics and report files to your machine for offline analysis.

## 🚨 Benchmarking Llama3 on Amazon Bedrock 🚨

Llama3 is now available on Bedrock (read [blog post](https://aws.amazon.com/blogs/aws/metas-llama-3-models-are-now-available-in-amazon-bedrock/)), and you can now benchmark it using `FMBench`. Here is the config file for benchmarking `Llama3-8b-instruct` and `Llama3-70b-instruct` on Bedrock.

<!-- markdown-link-check-disable -->
- [Config file](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/main/src/fmbench/configs/bedrock/config-bedrock-llama3.yml) for `Llama3-8b-instruct` and `Llama3-70b-instruct`.
<!-- markdown-link-check-enable -->

## 🚨 Benchmarking Llama3 on Amazon SageMaker 🚨

Llama3 is now available on SageMaker (read [blog post](https://aws.amazon.com/blogs/machine-learning/meta-llama-3-models-are-now-available-in-amazon-sagemaker-jumpstart/)), and you can now benchmark it using `FMBench`. Here are the config files for benchmarking `Llama3-8b-instruct` and `Llama3-70b-instruct` on `ml.p4d.24xlarge`, `ml.inf2.24xlarge` and `ml.g5.12xlarge` instances.

<!-- markdown-link-check-disable -->
- [Config file](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/main/src/fmbench/configs/llama3/8b/config-llama3-8b-instruct-g5-p4d.yml) for `Llama3-8b-instruct` on  `ml.p4d.24xlarge` and `ml.g5.12xlarge`.
- [Config file](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/main/src/fmbench/configs/llama3/70b/config-llama3-70b-instruct-g5-p4d.yml) for `Llama3-70b-instruct` on  `ml.p4d.24xlarge` and `ml.g5.48xlarge`.
- [Config file](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/main/src/fmbench/configs/llama3/8b/config-llama3-8b-inf2-g5.yml) for `Llama3-8b-instruct` on  `ml.inf2.24xlarge` and `ml.g5.12xlarge`.
<!-- markdown-link-check-enable -->

## Benchmarking Llama2 on Amazon SageMaker

Llama2 models are available through SageMaker JumpStart as well as directly deployable from Hugging Face to a SageMaker endpoint. You can use `FMBench` to benchmark Llama2 on SageMaker for different combinations of instance types and inference containers.

<!-- markdown-link-check-disable -->
- [Config file](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/main/src/fmbench/configs/llama2/7b/config-llama2-7b-g5-quick.yml) for `Llama2-7b` on `ml.g5.xlarge` and `ml.g5.2xlarge` instances, using the [Hugging Face TGI container](763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.0.1-tgi1.1.0-gpu-py39-cu118-ubuntu20.04).
- [Config file](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/main/src/fmbench/configs/llama2/7b/config-llama2-7b-g4dn-g5-trt.yml) for `Llama2-7b` on `ml.g4dn.12xlarge` instance using the [Deep Java Library DeepSpeed container](763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.26.0-deepspeed0.12.6-cu121).
- [Config file](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/main/src/fmbench/configs/llama2/13b/config-llama2-13b-inf2-g5-p4d.yml) for `Llama2-13b` on `ml.g5.12xlarge`, `ml.inf2.24xlarge` and `ml.p4d.24xlarge` instances using the [Hugging Face TGI container](763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.0.1-tgi1.1.0-gpu-py39-cu118-ubuntu20.04) and the [Deep Java Library & NeuronX container](763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.26.0-neuronx-sdk2.16.0).
- [Config file](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/main/src/fmbench/configs/llama2/70b/config-llama2-70b-g5-p4d-trt.yml) for `Llama2-70b` on `ml.p4d.24xlarge` instance using the [Deep Java Library TensorRT container](763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.26.0-tensorrtllm0.7.1-cu122).
- [Config file](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/main/src/fmbench/configs/llama2/70b/config-llama2-70b-inf2-g5.yml) for `Llama2-70b` on `ml.inf2.48xlarge` instance using the [HuggingFace TGI with Optimum NeuronX container](763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-tgi-inference:1.13.1-optimum0.0.17-neuronx-py310-ubuntu22.04).
<!-- markdown-link-check-enable -->

## Benchmarking Llama2 on Amazon Bedrock

The Llama2-13b-chat and Llama2-70b-chat models are available on [Bedrock](https://aws.amazon.com/bedrock/llama/). You can use `FMBench` to benchmark Llama2 on Bedrock for both on-demand throughput and provisioned throughput inference options.

<!-- markdown-link-check-disable -->
- [Config file](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/main/src/fmbench/configs/bedrock/config-bedrock.yml) for `Llama2-13b-chat` and `Llama2-70b-chat` on Bedrock for on-demand throughput.
<!-- markdown-link-check-enable -->

- For testing provisioned throughput simply replace the `ep_name` parameter in `experiments` section of the config file with the ARN of your provisioned throughput.

## More..

For bug reports, enhancement requests and any questions please create a [GitHub issue](https://github.com/aws-samples/foundation-model-benchmarking-tool/issues) on the `FMBench` repo.
