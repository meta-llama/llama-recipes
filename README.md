# Llama Recipes: Examples to get started using the Llama models from Meta
<!-- markdown-link-check-disable -->
The 'llama-recipes' repository is a companion to the [Meta Llama](https://github.com/meta-llama/llama-models) models. We support the latest version, [Llama 3.2 Vision](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md) and [Llama 3.2 Text](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md), in this repository. This repository contains example scripts and notebooks to get started with the models in a variety of use-cases, including fine-tuning for domain adaptation and building LLM-based applications with Llama and other tools in the LLM ecosystem. The examples here use Llama locally, in the cloud, and on-prem.

> [!TIP]
> Get started with Llama 3.2 with these new recipes:
> * [Finetune Llama 3.2 Vision](https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/finetuning/finetune_vision_model.md)
> * [Multimodal Inference with Llama 3.2 Vision](https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/inference/local_inference/README.md#multimodal-inference)
> * [Inference on Llama Guard 1B + Multimodal inference on Llama Guard 11B-Vision](https://github.com/meta-llama/llama-recipes/blob/main/recipes/responsible_ai/llama_guard/llama_guard_text_and_vision_inference.ipynb)


<!-- markdown-link-check-enable -->
> [!NOTE]
> Llama 3.2 follows the same prompt template as Llama 3.1, with a new special token `<|image|>` representing the input image for the multimodal models.
> 
> More details on the prompt templates for image reasoning, tool-calling and code interpreter can be found [on the documentation website](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_2).



## Table of Contents

- [Llama Recipes: Examples to get started using the Llama models from Meta](#llama-recipes-examples-to-get-started-using-the-llama-models-from-meta)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
      - [PyTorch Nightlies](#pytorch-nightlies)
    - [Installing](#installing)
      - [Install with pip](#install-with-pip)
      - [Install with optional dependencies](#install-with-optional-dependencies)
      - [Install from source](#install-from-source)
    - [Getting the Llama models](#getting-the-llama-models)
      - [Model conversion to Hugging Face](#model-conversion-to-hugging-face)
  - [Repository Organization](#repository-organization)
    - [`recipes/`](#recipes)
    - [`src/`](#src)
  - [Supported Features](#supported-features)
  - [Contributing](#contributing)
  - [License](#license)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

#### PyTorch Nightlies
If you want to use PyTorch nightlies instead of the stable release, go to [this guide](https://pytorch.org/get-started/locally/) to retrieve the right `--extra-index-url URL` parameter for the `pip install` commands on your platform.

### Installing
Llama-recipes provides a pip distribution for easy install and usage in other projects. Alternatively, it can be installed from source.

> [!NOTE]
> Ensure you use the correct CUDA version (from `nvidia-smi`) when installing the PyTorch wheels. Here we are using 11.8 as `cu118`.
> H100 GPUs work better with CUDA >12.0

#### Install with pip
```
pip install llama-recipes
```

#### Install with optional dependencies
Llama-recipes offers the installation of optional packages. There are three optional dependency groups.
To run the unit tests we can install the required dependencies with:
```
pip install llama-recipes[tests]
```
For the vLLM example we need additional requirements that can be installed with:
```
pip install llama-recipes[vllm]
```
To use the sensitive topics safety checker install with:
```
pip install llama-recipes[auditnlg]
```
Some recipes require the presence of langchain. To install the packages follow the recipe description or install with:
```
pip install llama-recipes[langchain]
```
Optional dependencies can also be combined with [option1,option2].

#### Install from source
To install from source e.g. for development use these commands. We're using hatchling as our build backend which requires an up-to-date pip as well as setuptools package.
```
git clone git@github.com:meta-llama/llama-recipes.git
cd llama-recipes
pip install -U pip setuptools
pip install -e .
```
For development and contributing to llama-recipes please install all optional dependencies:
```
git clone git@github.com:meta-llama/llama-recipes.git
cd llama-recipes
pip install -U pip setuptools
pip install -e .[tests,auditnlg,vllm]
```


### Getting the Llama models
You can find Llama models on Hugging Face hub [here](https://huggingface.co/meta-llama), **where models with `hf` in the name are already converted to Hugging Face checkpoints so no further conversion is needed**. The conversion step below is only for original model weights from Meta that are hosted on Hugging Face model hub as well.

#### Model conversion to Hugging Face
If you have the model checkpoints downloaded from the Meta website, you can convert it to the Hugging Face format with:

```bash
## Install Hugging Face Transformers from source
pip freeze | grep transformers ## verify it is version 4.45.0 or higher

git clone git@github.com:huggingface/transformers.git
cd transformers
pip install protobuf
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
   --input_dir /path/to/downloaded/llama/weights --model_size 3B --output_dir /output/path
```



## Repository Organization
Most of the code dealing with Llama usage is organized across 2 main folders: `recipes/` and `src/`.

### `recipes/`

Contains examples organized in folders by topic:
| Subfolder | Description |
|---|---|
[quickstart](./recipes/quickstart) | The "Hello World" of using Llama, start here if you are new to using Llama.
[use_cases](./recipes/use_cases)|Scripts showing common applications of Meta Llama3
[3p_integrations](./recipes/3p_integrations)|Partner owned folder showing common applications of Meta Llama3
[responsible_ai](./recipes/responsible_ai)|Scripts to use PurpleLlama for safeguarding model outputs
[experimental](./recipes/experimental)|Meta Llama implementations of experimental LLM techniques

### `src/`

Contains modules which support the example recipes:
| Subfolder | Description |
|---|---|
| [configs](src/llama_recipes/configs/) | Contains the configuration files for PEFT methods, FSDP, Datasets, Weights & Biases experiment tracking. |
| [datasets](src/llama_recipes/datasets/) | Contains individual scripts for each dataset to download and process. Note |
| [inference](src/llama_recipes/inference/) | Includes modules for inference for the fine-tuned models. |
| [model_checkpointing](src/llama_recipes/model_checkpointing/) | Contains FSDP checkpoint handlers. |
| [policies](src/llama_recipes/policies/) | Contains FSDP scripts to provide different policies, such as mixed precision, transformer wrapping policy and activation checkpointing along with any precision optimizer (used for running FSDP with pure bf16 mode). |
| [utils](src/llama_recipes/utils/) | Utility files for:<br/> - `train_utils.py` provides training/eval loop and more train utils.<br/> - `dataset_utils.py` to get preprocessed datasets.<br/> - `config_utils.py` to override the configs received from CLI.<br/> - `fsdp_utils.py` provides FSDP  wrapping policy for PEFT methods.<br/> - `memory_utils.py` context manager to track different memory stats in train loop. |


## Supported Features
The recipes and modules in this repository support the following features:

| Feature                                        |   |
| ---------------------------------------------- | - |
| HF support for inference                       | ✅ |
| HF support for finetuning                      | ✅ |
| PEFT                                           | ✅ |
| Deferred initialization ( meta init)           | ✅ |
| Low CPU mode for multi GPU                     | ✅ |
| Mixed precision                                | ✅ |
| Single node quantization                       | ✅ |
| Flash attention                                | ✅ |
| Activation checkpointing FSDP                  | ✅ |
| Hybrid Sharded Data Parallel (HSDP)            | ✅ |
| Dataset packing & padding                      | ✅ |
| BF16 Optimizer (Pure BF16)                     | ✅ |
| Profiling & MFU tracking                       | ✅ |
| Gradient accumulation                          | ✅ |
| CPU offloading                                 | ✅ |
| FSDP checkpoint conversion to HF for inference | ✅ |
| W&B experiment tracker                         | ✅ |


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License
<!-- markdown-link-check-disable -->

See the License file for Meta Llama 3.2 [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) and Acceptable Use Policy [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/USE_POLICY.md)

See the License file for Meta Llama 3.1 [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) and Acceptable Use Policy [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/USE_POLICY.md)

See the License file for Meta Llama 3 [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3/LICENSE) and Acceptable Use Policy [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3/USE_POLICY.md)

See the License file for Meta Llama 2 [here](https://github.com/meta-llama/llama-models/blob/main/models/llama2/LICENSE) and Acceptable Use Policy [here](https://github.com/meta-llama/llama-models/blob/main/models/llama2/USE_POLICY.md)
<!-- markdown-link-check-enable -->

## Supported Input Formats

- **PDF Documents**: Ingest and process text from PDF files.
- **Websites**: Extract and process text content from web URLs.
- **YouTube Videos**: Retrieve and transcribe audio from YouTube video URLs.
- **Audio Files**: Transcribe audio files into text using Whisper.

## Usage Examples

### Ingest from a PDF

```python
from ingestion import IngestorFactory

input_type = "pdf"
pdf_path = './resources/2402.13116v3.pdf'
extracted_text = ingest_content(input_type, pdf_path)
if extracted_text:
    with open('extracted_text.txt', 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    print("Extracted text has been saved to extracted_text.txt")
```

### Ingest from a Website

```python
from ingestion import IngestorFactory

input_type = "website"
website_url = "https://www.example.com"
website_text = ingest_content(input_type, website_url)
if website_text:
    with open('website_extracted_text.txt', 'w', encoding='utf-8') as f:
        f.write(website_text)
    print("Extracted website text has been saved to website_extracted_text.txt")
```

### Ingest from a YouTube Video

```python
from ingestion import IngestorFactory

input_type = "youtube"
youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
youtube_transcript = ingest_content(input_type, youtube_url)
if youtube_transcript:
    with open('youtube_transcript.txt', 'w', encoding='utf-8') as f:
        f.write(youtube_transcript)
    print("YouTube transcript has been saved to youtube_transcript.txt")
```

### Ingest from an Audio File

```python
from ingestion import IngestorFactory

input_type = "audio"
audio_file = './resources/sample_audio.mp3'
audio_transcription = ingest_content(input_type, audio_file, model_type="base")
if audio_transcription:
    with open('audio_transcription.txt', 'w', encoding='utf-8') as f:
        f.write(audio_transcription)
    print("Audio transcription has been saved to audio_transcription.txt")
```

## Step 4: Testing

Ensure that each ingestor works as expected by testing with sample inputs.

### 4.1. Create Test Cases

```python
# test_ingestion.py

import unittest
from ingestion import IngestorFactory

class TestIngestion(unittest.TestCase):

    def test_pdf_ingestion(self):
        pdf_path = "./resources/sample.pdf"
        ingestor = IngestorFactory.get_ingestor("pdf")
        text = ingestor.extract_text(pdf_path)
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    def test_website_ingestion(self):
        website_url = "https://www.example.com"
        ingestor = IngestorFactory.get_ingestor("website")
        text = ingestor.extract_text(website_url)
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    def test_youtube_ingestion(self):
        youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        ingestor = IngestorFactory.get_ingestor("youtube")
        transcript = ingestor.extract_text(youtube_url)
        self.assertIsInstance(transcript, str)
        self.assertTrue(len(transcript) > 0)

    def test_audio_ingestion(self):
        audio_file = "./resources/sample_audio.mp3"
        ingestor = IngestorFactory.get_ingestor("audio", model_type="base")
        transcription = ingestor.extract_text(audio_file)
        self.assertIsInstance(transcription, str)
        self.assertTrue(len(transcription) > 0)
    
    def test_unsupported_type(self):
        ingestor = IngestorFactory.get_ingestor("unsupported")
        self.assertIsNone(ingestor)

if __name__ == "__main__":
    unittest.main()
```

### 4.2. Run Tests

Execute the tests to verify all ingestion methods function correctly.

```bash
python test_ingestion.py
```

Ensure all tests pass and handle any exceptions or errors that arise.

## Conclusion

By following these steps, you've successfully **extended your `ingestion.py` module** to support multiple input formats—**websites, YouTube links, and audio files**—in addition to PDFs. This enhancement broadens the usability of your `NotebookLlama` pipeline, making it more versatile and valuable.

### Next Steps

1. **Handle Edge Cases**: Enhance each ingestor to manage various edge cases, such as unsupported formats, network issues, or transcription errors.
2. **Asynchronous Processing**: Implement asynchronous ingestion to improve pipeline efficiency, especially for time-consuming tasks like audio transcription.
3. **Logging and Error Reporting**: Integrate comprehensive logging to monitor ingestion processes and facilitate troubleshooting.
4. **User Interface Enhancements**: Improve the interactive widgets in your notebook to provide better feedback and progress indicators during ingestion.
5. **Documentation**: Continue to refine your documentation with detailed explanations, troubleshooting tips, and advanced usage examples.

Feel free to reach out if you need further assistance or have more features you'd like to implement. Happy coding!
