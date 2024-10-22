### Ideas: NotebookLLama

Steps:  
Path:

1. Decide the Topic
- Upload a PDF
OR - Put in a topic -> Scraped 
- Report written 

2. 2 Agents debate/interact? Podcast style -> Write a Transcript

3. TTS Engine (E25 or ) Make the podcast

### Instructions: 

Running 1B-Model: ```python 1B-chat-start.py --temperature 0.7 --top_p 0.9 --system_message "you are acting as an old angry uncle and will debate why LLMs are bad" --user_message "I love LLMs"```

Running Debator: ```python 1B-debating-script.py --initial_topic "The future of space exploration" --system_prompt1 "You are an enthusiastic advocate for space exploration" --system_prompt2 "You are a skeptic who believes we should focus on Earth's problems first" --n_turns 4 --temperature 0.8 --top_p 0.9 --model_name "meta-llama/Llama-3.2-1B-Instruct"```

### Scratch-pad/Running Notes:

Bark is cool but just v6 works great, I tried v9 but its quite robotic and that is sad. 

So Parler is next-its quite cool for prompting 

xTTS-v2 by coquai is cool, however-need to check the license-I think an example is allowed

Torotoise is blocking because it needs HF version that doesnt work with llama-3.2 models so I will probably need to make a seperate env-need to eval if its worth it

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

Starting with: Bark but if it falls apart, here is the order

- 0: https://huggingface.co/suno/bark
- 1: https://huggingface.co/WhisperSpeech/WhisperSpeech
- 2: https://huggingface.co/spaces/parler-tts/parler_tts


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

### Resources I used to learn about Suno:

- https://betterprogramming.pub/text-to-audio-generation-with-bark-clearly-explained-4ee300a3713a
- https://colab.research.google.com/drive/1dWWkZzvu7L9Bunq9zvD-W02RFUXoW-Pd?usp=sharing
- https://colab.research.google.com/drive/1eJfA2XUa-mXwdMy7DoYKVYHI1iTd9Vkt?usp=sharing#scrollTo=NyYQ--3YksJY
- https://replicate.com/suno-ai/bark?prediction=zh8j6yddxxrge0cjp9asgzd534

