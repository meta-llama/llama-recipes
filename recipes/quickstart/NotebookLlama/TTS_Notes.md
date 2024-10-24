### Notes from TTS Experimentation

For the TTS Pipeline, *all* of the top models from HuggingFace and Reddit were tried. 

The goal was to use the models that were easy to setup and sounded less robotic with ability to include sound effects like laughter, etc.

#### Parler-TTS



Surprisingly, Parler's mini model sounded more natural. In their [repo]() they share names of speakers that we can use in prompt 

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

#### Suno/Bark

Bark is cool but just v6 works great, I tried v9 but its quite robotic and that is sad. 

Bark-Tests: Best results for speaker/v6 are at ```speech_output = model.generate(**inputs, temperature = 0.9, semantic_temperature = 0.8)
Audio(speech_output[0].cpu().numpy(), rate=sampling_rate)```

Tested sound effects:

- Laugh is probably most effective
- Sigh is hit or miss
- Gasps doesn't work
- A singly hypen is effective
- Captilisation makes it louder


### Notes from other models that were tested:

Promising directions to explore in future:

- [MeloTTS](huggingface.co/myshell-ai/MeloTTS-English) This is most popular (ever) on HuggingFace
- [WhisperSpeech](https://huggingface.co/WhisperSpeech/WhisperSpeech) sounded quite natural as well
- [F5-TTS](https://github.com/SWivid/F5-TTS) was the latest release at this time, however, it felt a bit robotic
- E2-TTS: r/locallama claims this to be a little better, however, it didn't pass the vibe test
- [xTTS](https://coqui.ai/blog/tts/open_xtts) It has great documentation and also seems promising



#### Some more models that weren't tested:

In other words, we leave this as an excercise to readers :D

- [Fish-Speech](https://huggingface.co/fishaudio/fish-speech-1.4)
- [MMS-TTS-Eng](https://huggingface.co/facebook/mms-tts-eng)
- [Metavoice](https://huggingface.co/metavoiceio/metavoice-1B-v0.1)
- [Hifigan](https://huggingface.co/nvidia/tts_hifigan)
- [TTS-Tacotron2](https://huggingface.co/speechbrain/tts-tacotron2-ljspeech) 
- [MMS-TTS-Eng](https://huggingface.co/facebook/mms-tts-eng)
- [VALL-E X](https://github.com/Plachtaa/VALL-E-X)
