Below are recipes for deploying common Llama workflows on [Crusoe's](https://crusoe.ai) high-performance, sustainable cloud. Each workflow corresponds to a subfolder with its own README and supplemental materials. Please reference the table below for hardware requirements.

| Workflow | Model(s) | VM type | Storage |
|:----:  | :----:  | :----:| :----: |
| [Serving Llama3.1 in FP8 with vLLM](vllm-fp8/) | [meta-llama/Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct), [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) | l40s-48gb.8x | 256 GiB Persistent Disk |

# Requirements
First, ensure that you have a Crusoe account (you can sign up [here](https://console.crusoecloud.com/)). We will provision resources using Terraform, please ensure that your environment is configured and refer to the Crusoe [docs](https://github.com/crusoecloud/terraform-provider-crusoe?tab=readme-ov-file#getting-started) for guidance.

# Serving Models
Some recipes in this repo require firewall rules to expose ports in order to reach the inference server. To manage firewall rules, please refer to our [networking documentation](https://docs.crusoecloud.com/networking/firewall-rules/managing-firewall-rules).
