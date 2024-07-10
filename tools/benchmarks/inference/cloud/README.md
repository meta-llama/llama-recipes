# Llama-Cloud-API-Benchmark
This folder contains code to run inference benchmark for Llama 2 models on cloud API with popular cloud service providers. The benchmark will focus on overall inference **throughput** for querying the API endpoint for output generation with different level of concurrent requests. Remember that to send queries to the API endpoint, you are required to acquire subscriptions with the cloud service providers and there will be a fee associated with it.

Disclaimer - The purpose of the code is to provide a configurable setup to measure inference throughput. It is not a representative of the performance of these API services and we do not plan to make comparisons between different API providers.


# Azure - Getting Started
To get started, there are certain steps we need to take to deploy the models:

<!-- markdown-link-check-disable -->
* Register for a valid Azure account with subscription [here](https://azure.microsoft.com/en-us/free/search/?ef_id=_k_CjwKCAiA-P-rBhBEEiwAQEXhH5OHAJLhzzcNsuxwpa5c9EJFcuAjeh6EvZw4afirjbWXXWkiZXmU2hoC5GoQAvD_BwE_k_&OCID=AIDcmm5edswduu_SEM__k_CjwKCAiA-P-rBhBEEiwAQEXhH5OHAJLhzzcNsuxwpa5c9EJFcuAjeh6EvZw4afirjbWXXWkiZXmU2hoC5GoQAvD_BwE_k_&gad_source=1&gclid=CjwKCAiA-P-rBhBEEiwAQEXhH5OHAJLhzzcNsuxwpa5c9EJFcuAjeh6EvZw4afirjbWXXWkiZXmU2hoC5GoQAvD_BwE)
<!-- markdown-link-check-enable -->
* Take a quick look on what is the [Azure AI Studio](https://learn.microsoft.com/en-us/azure/ai-studio/what-is-ai-studio?tabs=home) and navigate to the website from the link in the article
* Follow the demos in the article to create a project and [resource](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal) group, or you can also follow the guide [here](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-llama?tabs=azure-studio)
* Select Llama models from Model catalog
* Click the "Deploy" button
* Select Serverless API with Azure AI Content Safety. Note that currently this API service is offered for Llama 2 pretrained model, chat model and Llama 3 instruct model
* Select the project you created in previous step
* Choose a deployment name then Go to deployment

Once deployed successfully, you should be assigned for an API endpoint and a security key for inference.
For more information, you should consult Azure's official documentation [here](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-llama?tabs=azure-studio) for model deployment and inference.

Now, replace the endpoint url and API key in ```azure/parameters.json```. For parameter `MODEL_ENDPOINTS`, with chat models the suffix should be `v1/chat/completions` and with pretrained models the suffix should be `v1/completions`.
Note that the API endpoint might implemented a rate limit for token generation in certain amount of time. If you encountered the error, you can try reduce `MAX_NEW_TOKEN` or start with smaller `CONCURRENT_LEVELS`.

For `MODEL_PATH`, copy the model path from Huggingface under meta-llama organization. For Llama 2, make sure you copy the path of the model with hf format. This model path is used to retrieve corresponding tokenizer for your model of choice. Llama 3 used a different tokenizer compare to Llama 2.

Once everything configured, to run chat model benchmark:
```python chat_azure_api_benchmark.py```

To run pretrained model benchmark:
```python pretrained_azure_api_benchmark.py```

Once finished, the result will be written into a CSV file in the same directory, which can be later imported into dashboard of your choice.
