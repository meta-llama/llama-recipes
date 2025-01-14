## NotebookLlama: An Open Source version of NotebookLM

![NotebookLlama](./resources/Outline.jpg)

[Listen to audio from the example here](./resources/_podcast.mp3)

This is a guided series of tutorials/notebooks that can be taken as a reference or course to build a PDF to Podcast workflow. 

You will also learn from the experiments of using Text to Speech Models.

It assumes zero knowledge of LLMs, prompting, and audio models; everything is covered in their respective notebooks.

### Outline:

Here is a step-by-step guide for the task:

- **Step 1: Pre-process PDF**: Use [`Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) to pre-process the PDF and save it in a `.txt` file.
- **Step 2: Transcript Writer**: Use [`Llama-3.1-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) model to write a podcast transcript from the text.
- **Step 3: Dramatic Re-Writer**: Use [`Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model to make the transcript more dramatic.
- **Step 4: Text-To-Speech Workflow**: Use `parler-tts/parler-tts-mini-v1` and `bark/suno` to generate a conversational podcast.

**Note 1**: In Step 1, we prompt the `Llama-3.2-1B-Instruct` model to not modify or summarize the text but strictly clean up extra characters or garbage characters that might get picked up due to encoding from the PDF. Please see the prompt in [Notebook 1: Pre-process PDF](Notebook1_PreprocessPDF.ipynb) for more details.

**Note 2**: For Step 2, you can also use the `Llama-3.1-8B-Instruct` model. We recommend experimenting to see if you observe any differences. The 70B model was used here because it provided slightly more creative podcast transcripts in our tests.

**Note 3**: For Step 4, please try to extend the approach with other models. These models were chosen based on sample prompts and worked best. Newer models might sound better. Please see [Notes](./TTS_Notes.md) for some sample tests.

### Detailed steps on running the notebook:

**Requirements**: 

- **GPU Server**: Required for using 70B, 8B, and 1B Llama models.
- **70B Model**: Requires a GPU with approximately 140GB of aggregated memory to infer in bfloat-16 precision.

**Note**: If you do not have access to high-memory GPUs, you can use the 8B and lower models for the entire pipeline without significant loss in functionality.

- **Login to Hugging Face**: Make sure to login using the `huggingface cli` and then launch your Jupyter notebook server to ensure you can download the Llama models.

  You'll need your Hugging Face access token, which you can obtain from your [Settings page](https://huggingface.co/settings/tokens). Then run `huggingface-cli login` and paste your Hugging Face access token to complete the login, ensuring the scripts can download Hugging Face models as needed.

- **Install Requirements**:

  Clone the repository and install dependencies by running the following commands inside the folder:

  ```bash
  git clone https://github.com/meta-llama/llama-recipes
  cd llama-recipes/recipes/quickstart/NotebookLlama/
  pip install -r requirements.txt
  ```

- **Notebook 1: Pre-process PDF** (`Notebook1_PreprocessPDF.ipynb`):

  This notebook processes the PDF and converts it into a `.txt` file using the new Feather light model.
  
  - Update the first cell with a PDF link that you would like to use. Ensure the link is correct before running the notebook.
  - Experiment with the prompts for the `Llama-3.2-1B-Instruct` model to improve results.

- **Notebook 2: Transcript Writer** (`Notebook2_TranscriptWriter.ipynb`):

  This notebook takes the processed output from Notebook 1 and generates a podcast transcript using the `Llama-3.1-70B-Instruct` model. If you have ample GPU resources, feel free to test with the 405B model!
  
  - Experiment with system prompts to improve results.
  - Try using the 8B model to compare differences.

- **Notebook 3: Dramatic Re-Writer** (`Notebook3_DramaticReWriter.ipynb`):

  This notebook enhances the transcript by adding dramatization and interruptions using the `Llama-3.1-8B-Instruct` model.
  
  - The notebook returns a tuple of conversations, simplifying subsequent steps.
  - Experiment with system prompts to further improve results.
  - Consider testing with the feather light 3B and 1B models.

- **Notebook 4: Text-To-Speech Workflow** (`Notebook4_TextToSpeechWorkflow.ipynb`):

  Convert the enhanced transcript into a podcast using `parler-tts/parler-tts-mini-v1` and `bark/suno` models.
  
  - The speakers and prompts for the parler model were chosen based on experimentation and suggestions from model authors.
  - Experiment with different TTS models and prompts to improve the natural sound of the podcast.

#### Note: Currently, there is an issue where Parler requires `transformers` version 4.43.3 or earlier, conflicting with steps 1-3. In Notebook 4, we switch the `transformers` version to accommodate Parler. Ensure you follow the notebook's instructions carefully to avoid dependency conflicts.

### Next Improvements & Further Ideas:

- **Speech Model Experimentation**: Improve the naturalness of the podcast by experimenting with different TTS models.
- **LLM vs. LLM Debate**: Utilize two agents to debate the topic of interest and generate the podcast outline.
- **Testing 405B Model**: Assess performance differences when using the 405B model for writing transcripts.
- **Enhanced Prompting**: Refine system prompts for improved results.
- **Support for Additional Input Sources**: Enable ingestion of websites, audio files, YouTube links, etc. Community contributions are welcome!

### Resources for Further Learning:

- [Text to Audio Generation with Bark - Clearly Explained](https://betterprogramming.pub/text-to-audio-generation-with-bark-clearly-explained-4ee300a3713a)
- [Colab Notebook for Text Processing](https://colab.research.google.com/drive/1dWWkZzvu7L9Bunq9zvD-W02RFUXoW-Pd?usp=sharing)
- [Replicate: Bark Model](https://replicate.com/suno-ai/bark?prediction=zh8j6yddxxrge0cjp9asgzd534)
- [Suno AI Notion Page](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c)

### Supported Input Sources:

NotebookLlama supports multiple input formats:

- **PDF files** (`*.pdf`)
- **Web pages** (`http://`, `https://`)
- **YouTube videos** (`youtube.com`, `youtu.be`)

To use a different input source, simply provide the appropriate path or URL when running the notebooks.
