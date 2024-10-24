### NotebookLlama: An Open Source version of NotebookLM

Author: Sanyam Bhutani

This is a guided series of tutorials/notebooks that can be taken as a reference or course to build a PDF to Podcast workflow.

It assumes zero knowledge of LLMs, prompting and audio models, everything is covered in their respective notebooks.

#### Outline:

Requirements: GPU server or an API provider for using 70B, 8B and 1B Llama models.

Note: For our GPU Poor friends, you can also use the 8B and lower models for the entire pipeline. There is no strong recommendation. The pipeline below is what worked best on first few tests. You should try and see what works best for you!

Here is step by step (pun intended) thought for the task:

- Step 1: Pre-process PDF: Use `Llama-3.2-1B` to pre-process and save a PDF
- Step 2: Transcript Writer: Use `Llama-3.1-70B` model to write a podcast transcript from the text
- Step 3: Dramatic Re-Writer: Use `Llama-3.1-8B` model to make the transcript more dramatic
- Step 4: Text-To-Speech Workflow: Use `parler-tts/parler-tts-mini-v1` and `bark/suno` to generate a conversational podcast

### Steps to running the notebook:

- Install the requirements from [here]() by running inside the folder:

```
git clone 
cd 
pip install -r requirements.txt
```

- Decide on a PDF to use for Notebook 1, it can be any link but please remember to update the first cell of the notebook with the right link

- 


So right now there is one issue: Parler needs transformers 4.43.3 or earlier and to generate you need latest, so I am just switching on fly in the notebooks.

TODO-MORE

### Next-Improvements/Further ideas:

- Speech Model experimentation: The TTS model is the limitation of how natural this will sound. This probably be improved with a better pipeline
- LLM vs LLM Debate: Another approach of writing the podcast would be having two agents debate the topic of interest and write the podcast outline. Right now we use a single LLM (70B) to write the podcast outline
- Testing 405B for writing the transcripts
- Better prompting
- Support for ingesting a website, audio file, YouTube links and more. We welcome community PRs!

### Scratch-pad/Running Notes:

Actually this IS THE MOST CONSISTENT PROMPT:
Small:
```
description = """
Laura's voice is expressive and dramatic in delivery, speaking at a fast pace with a very close recording that almost has no background noise.
"""
```

Large: 
```
description = """
Alisa's voice is consistent, quite expressive and dramatic in delivery, with a very close recording that almost has no background noise.
"""
```
Small:
```
description = """
Jenna's voice is consistent, quite expressive and dramatic in delivery, with a very close recording that almost has no background noise.
"""
```

Bark is cool but just v6 works great, I tried v9 but its quite robotic and that is sad. 

So Parler is next-its quite cool for prompting 

xTTS-v2 by coquai is cool, however-need to check the license-I think an example is allowed

Torotoise is blocking because it needs HF version that doesnt work with llama-3.2 models so I will probably need to make a seperate env-need to eval if its worth it

Side note: The TTS library is a really cool effort!

Bark-Tests: Best results for speaker/v6 are at ```speech_output = model.generate(**inputs, temperature = 0.9, semantic_temperature = 0.8)
Audio(speech_output[0].cpu().numpy(), rate=sampling_rate)```

Tested sound effects:

- Laugh is probably most effective
- Sigh is hit or miss
- Gasps doesn't work
- A singly hypen is effective
- Captilisation makes it louder

Ignore/Delete this in final stages, right now this is a "vibe-check" for TTS model(s):

- https://github.com/SWivid/F5-TTS: Latest and most popular-"feels robotic"
- Reddit says E2 model from earlier is better

S
- 1: https://huggingface.co/WhisperSpeech/WhisperSpeech


Vibe check: 
- This is most popular (ever) on HF and features different accents-the samples feel a little robotic and no accent difference: https://huggingface.co/myshell-ai/MeloTTS-English
- Seems to have great documentation but still a bit robotic for my liking: https://coqui.ai/blog/tts/open_xtts
- Super easy with laughter etc but very slightly robotic: https://huggingface.co/suno/bark
- This is THE MOST NATURAL SOUNDING: https://huggingface.co/WhisperSpeech/WhisperSpeech
- This has a lot of promise, even though its robotic, we can use natural voice to add filters or effects: https://huggingface.co/spaces/parler-tts/parler_tts

Higher Barrier to testing (In other words-I was too lazy to test):
- https://huggingface.co/fishaudio/fish-speech-1.4
- https://huggingface.co/facebook/mms-tts-eng
- https://huggingface.co/metavoiceio/metavoice-1B-v0.1
- https://huggingface.co/nvidia/tts_hifigan
- https://huggingface.co/speechbrain/tts-tacotron2-ljspeech


Try later:
- Whisper Colab: 
- https://huggingface.co/parler-tts/parler-tts-large-v1
- https://huggingface.co/myshell-ai/MeloTTS-English
- Bark: https://huggingface.co/suno/bark (This has been insanely popular)
- https://huggingface.co/facebook/mms-tts-eng
- https://huggingface.co/fishaudio/fish-speech-1.4
- https://huggingface.co/mlx-community/mlx_bark
- https://huggingface.co/metavoiceio/metavoice-1B-v0.1
- https://huggingface.co/suno/bark-small

### Resources for further learning:

- https://betterprogramming.pub/text-to-audio-generation-with-bark-clearly-explained-4ee300a3713a
- https://colab.research.google.com/drive/1dWWkZzvu7L9Bunq9zvD-W02RFUXoW-Pd?usp=sharing
- https://colab.research.google.com/drive/1eJfA2XUa-mXwdMy7DoYKVYHI1iTd9Vkt?usp=sharing#scrollTo=NyYQ--3YksJY
- https://replicate.com/suno-ai/bark?prediction=zh8j6yddxxrge0cjp9asgzd534

