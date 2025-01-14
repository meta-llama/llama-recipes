### Notes from TTS Experimentation

For the TTS Pipeline, *all* of the top models from HuggingFace and Reddit were tried. 

The goal was to use the models that were easy to setup and sounded less robotic with ability to include sound effects like laughter, etc.

#### Parler-TTS

Minimal code to run their models:

```
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

# Define text and description
text_prompt = "This is where the actual words to be spoken go"
description = """
Laura's voice is expressive and dramatic in delivery, speaking at a fast pace with a very close recording that almost has no background noise.
"""

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(text_prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()

ipd.Audio(audio_arr, rate=model.config.sampling_rate)
```

The really cool aspect of these models are the ability to prompt the `description` which can change the speaker profile and pacing of the outputs.

Surprisingly, Parler's mini model sounded more natural.

In their [repo](https://github.com/huggingface/parler-tts/blob/main/INFERENCE.md#speaker-consistency) they share names of speakers that we can use in prompt.

#### Suno/Bark

Minimal code to run bark:

```
voice_preset = "v2/en_speaker_6"
sampling_rate = 24000

text_prompt = """
Exactly! [sigh] And the distillation part is where you take a LARGE-model,and compress-it down into a smaller, more efficient model that can run on devices with limited resources.
"""
inputs = processor(text_prompt, voice_preset=voice_preset).to(device)

speech_output = model.generate(**inputs, temperature = 0.9, semantic_temperature = 0.8)
Audio(speech_output[0].cpu().numpy(), rate=sampling_rate)
```

Similar to parler models, suno has a [library](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c) of speakers.

v9 from their library sounded robotic so we use Parler for our first speaker and the best one from bark.

The incredible thing about Bark model is being able to add sound effects: `[Laugh]`, `[Gasps]`, `[Sigh]`, `[clears throat]`, making words capital causes the model to emphasize them. 

Adding `-` gives a break in the text. We utilize this knowledge when we re-write the transcript using the 8B model to add effects to our transcript.

Note: Authors suggest using `...`. However, this didn't work as effectively as adding a hyphen during trails.

#### Hyper-parameters: 

Bark models have two parameters we can tweak: `temperature` and `semantic_temperature`

Below are the notes from a sweep, prompt and speaker were fixed and this was a vibe test to see which gives best results. `temperature` and `semantic_temperature` respectively below:

First, fix `temperature` and sweep `semantic_temperature`
- `0.7`, `0.2`: Quite bland and boring
- `0.7`, `0.3`: An improvement over the previous one
- `0.7`, `0.4`: Further improvement 
- `0.7`, `0.5`: This one didn't work
- `0.7`, `0.6`: So-So, didn't stand out
- `0.7`, `0.7`: The best so far
- `0.7`, `0.8`: Further improvement 
- `0.7`, `0.9`: Mix feelings on this one

Now sweeping the `temperature`
- `0.1`, `0.9`: Very Robotic
- `0.2`, `0.9`: Less Robotic but not convincing
- `0.3`, `0.9`: Slight improvement still not fun
- `0.4`, `0.9`: Still has a robotic tinge
- `0.5`, `0.9`: The laugh was weird on this one but the voice modulates so much it feels speaker is changing
- `0.6`, `0.9`: Most consistent voice but has a robotic after-taste
- `0.7`, `0.9`: Very robotic and laugh was weird
- `0.8`, `0.9`: Completely ignore the laughter but it was more natural
- `0.9`, `0.9`: We have a winner probably

After this about ~30 more sweeps were done with the promising combinations:

Best results are at ```speech_output = model.generate(**inputs, temperature = 0.9, semantic_temperature = 0.8)
Audio(speech_output[0].cpu().numpy(), rate=sampling_rate)```


### Notes from other models that were tested:

Promising directions to explore in future:

- [MeloTTS](https://huggingface.co/myshell-ai/MeloTTS-English) This is most popular (ever) on HuggingFace
- [WhisperSpeech](https://huggingface.co/WhisperSpeech/WhisperSpeech) sounded quite natural as well
- [F5-TTS](https://github.com/SWivid/F5-TTS) was the latest release at this time, however, it felt a bit robotic
- E2-TTS: r/locallama claims this to be a little better, however, it didn't pass the vibe test
- [xTTS](https://coqui.ai/blog/tts/open_xtts) It has great documentation and also seems promising

#### Some more models that weren't tested:

In other words, we leave this as an exercise to readers :D

- [Fish-Speech](https://huggingface.co/fishaudio/fish-speech-1.4)
- [MMS-TTS-Eng](https://huggingface.co/facebook/mms-tts-eng)
- [Metavoice](https://huggingface.co/metavoiceio/metavoice-1B-v0.1)
- [Hifigan](https://huggingface.co/nvidia/tts_hifigan)
- [TTS-Tacotron2](https://huggingface.co/speechbrain/tts-tacotron2-ljspeech) 
- [MMS-TTS-Eng](https://huggingface.co/facebook/mms-tts-eng)
- [VALL-E X](https://github.com/Plachtaa/VALL-E-X)
