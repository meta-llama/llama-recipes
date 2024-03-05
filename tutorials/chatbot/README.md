#### Introduction

Large language models (LLMs) have emerged as groundbreaking tools, capable of understanding and generating human-like text. These models power many of today's advanced chatbots, providing more natural and engaging user experiences. But how do we create these intelligent systems? 

Here, we aim to make an FAQ model for Llama that be able to answer questions about Llama by fine-tune Llama2 7B chat using existing official Llama documents.


#### Fine-tuning Process

Fine-tuning LLMs here LLama 2 involves several key steps:

1. **Data Collection:** Gathering a diverse and comprehensive dataset is crucial. This dataset should include a wide range of topics and conversational styles to ensure the model can handle various subjects. A recent [research](https://arxiv.org/pdf/2305.11206.pdf) shows that quality of data has far more importance than quantity. Here are some high level thoughts on data collection:

- Source Identification: Identify the sources where your FAQs are coming from. This could include websites, customer service transcripts, emails, forums, and product manuals. Prioritize sources that reflect the real questions your users are asking.

- Diversity and Coverage: Ensure your data covers a wide range of topics relevant to your domain. It's crucial to include variations in how questions are phrased to make your model robust to different wording.

- Volume: The amount of data needed depends on the complexity of the task and the variability of the language in your domain. Generally, more data leads to a better-performing model, but aim for high-quality, relevant data.

Here, we are going to use [self-instruct](https://arxiv.org/abs/2212.10560) idea and use Llama model to build our dataset, for details please check this [doc](./data_pipelines/REAME.md).

2. **Data Formatting** 

For a FAQ model, you need to format your data in a way that's conducive to learning question-answer relationships. A common format is the question-answer (QA) pair:

Question-Answer Pairing: Organize your data into pairs where each question is directly followed by its answer. This simple structure is highly effective for training models to understand and generate responses. For example:

```python 
"question": "What is Llama 2?",
"answer": "Llama 2 is a collection of pretrained and fine-tuned large language models ranging from 7 billion to 70 billion parameters, optimized for dialogue use cases."
```


3. **Preprocessing:** This step involves cleaning the data and preparing it for training. It might include removing irrelevant information, correcting errors, and splitting the data into training and evaluation sets.


4. **Fine-Tuning:** Given that we have a selected pretrained model, in this case we use LLama 2 chat 7B, fine-tunning with more specific data can improve its performance on particular tasks, such as answering questions about Llama in this case.

