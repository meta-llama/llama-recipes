## [Running LLama2 on EC2](llama2_ec2.py)
This demo app runs `Llama-2-13b-chat-hf` from Hugging Face on an `A10G:1` on AWS EC2. It uses [Runhouse](https://github.com/run-house/runhouse) and [SkyPilot](https://github.com/skypilot-org/skypilot?tab=readme-ov-file) to provision a GPU, sync your code to it, install dependencies, and set up the function on a server that can be called remotely.

For more information about Runhouse, check out their [docs](https://www.run.house/docs) and other [examples](https://www.run.house/examples). For further questions, chat with them on [Discord](https://discord.com/invite/RnhB6589Hs) or file an [issue on GitHub](https://github.com/run-house/runhouse/issues).

Make sure your necessary credentials are setup: 

- Your Huggingface token (`export HF_TOKEN=...`)
- Your AWS credentials (`~/.aws/credentials`)

Make sure you have a general Llama conda env set up:
```
conda create -n llama-demo-apps python=3.8
conda activate llama-demo-apps
```

Then, run:
```
pip install "runhouse[sky]" transformers torch
sky check
```
Make sure `sky check` says AWS is enabled. Then:

```
git clone https://github.com/facebookresearch/llama-recipes
cd llama-recipes/recipes/inference/llama2_13b_ec2/
python llama2_ec2.py
```

This will have some more initial up time to set up the EC2 instance, but if you run the same code again it'll reuse the existing cluster. The instance is managed via SkyPilot; you can run `sky down rh-a10x` to bring down the EC2 instance.