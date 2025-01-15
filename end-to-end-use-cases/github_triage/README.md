# Automatic Issues Triaging with Llama

This tool utilizes an off-the-shelf Llama model to analyze, generate insights, and create a report for better understanding of the state of a repository. It serves as a reference implementation for using Llama to develop custom reporting and data analytics applications.

## Features

The tool performs the following tasks:

* Fetches issue threads from a specified repository
* Analyzes issue discussions and generates annotations such as category, severity, component affected, etc.
* Categorizes all issues by theme
* Synthesizes key challenges faced by users, along with probable causes and remediations
* Generates a high-level executive summary providing insights on diagnosing and improving the developer experience

For a step-by-step look, check out the [walkthrough notebook](walkthrough.ipynb).

## Getting Started


### Installation

```bash
pip install -r requirements.txt
```

### Setup

1. **API Keys and Model Service**: Set your GitHub token for API calls. Some privileged information may not be available if you don't have push-access to the target repository.
2. **Model Configuration**: Set the appropriate values in the `model` section of [config.yaml](config.yaml) for using Llama via VLLM or Groq.
3. **JSON Schemas**: Edit the output JSON schemas in [config.yaml](config.yaml) to ensure consistency in outputs. VLLM supports JSON-decoding via the `guided_json` generation argument, while Groq requires passing the schema in the system prompt.

### Running the Tool

```bash
python triage.py --repo_name='meta-llama/llama-recipes' --start_date='2024-08-14' --end_date='2024-08-27'
```

### Output

The tool generates:

* CSV files with `annotations`, `challenges`, and `overview` data, which can be persisted in SQL tables for downstream analyses and reporting.
* Graphical matplotlib plots of repository traffic, maintenance activity, and issue attributes.
* A PDF report for easier reading and sharing.

## Config

The tool's configuration is stored in [config.yaml](config.yaml). The following sections can be edited:

* **Github Token**: Use a token that has push-access on the target repo.
* **model**: Specify the model service (`vllm` or `groq`) and set the endpoints and API keys as applicable.
* **prompts**: For each of the 3 tasks Llama does in this tool, we specify a prompt and an output JSON schema:
  * `parse_issue`: Parsing and generating annotations for the issues 
  * `assign_category`: Assigns each issue to a category specified in an enum in the corresponding JSON schema
  * `get_overview`: Generates a high-level executive summary and analysis of all the parsed and generated data

## Troubleshooting

* If you encounter issues with API calls, ensure that your GitHub token is set correctly and that you have the necessary permissions.
* If you encounter issues with the model service, check the configuration values in [config.yaml](config.yaml).
