### Notes from TTS Experimentation

For the TTS Pipeline, *all* of the top models from HuggingFace and Reddit were tested. 

Tested how? 

It was a simple vibe test of checking which sounds less robotic. Promoising directions to explore in future:

- [MeloTTS](huggingface.co/myshell-ai/MeloTTS-English) This is most popular (ever) on HuggingFace
- [WhisperSpeech](https://huggingface.co/WhisperSpeech/WhisperSpeech) sounded quite natural as well
- 


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

Vibe check: 
- 
- Seems to have great documentation but still a bit robotic for my liking: https://coqui.ai/blog/tts/open_xtts

- This is THE MOST NATURAL SOUNDING: 
- This has a lot of promise, even though its robotic, we can use natural voice to add filters or effects: https://huggingface.co/spaces/parler-tts/parler_tts

Higher Barrier to testing (In other words-I was too lazy to test):
- https://huggingface.co/fishaudio/fish-speech-1.4
- https://huggingface.co/facebook/mms-tts-eng
- https://huggingface.co/metavoiceio/metavoice-1B-v0.1
- https://huggingface.co/nvidia/tts_hifigan
- https://huggingface.co/speechbrain/tts-tacotron2-ljspeech


Try later:
- Whisper Colab: 
- https://huggingface.co/facebook/mms-tts-eng
- https://huggingface.co/fishaudio/fish-speech-1.4
- https://huggingface.co/metavoiceio/metavoice-1B-v0.1