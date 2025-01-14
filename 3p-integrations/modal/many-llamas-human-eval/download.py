# ## Downloading Llama 3.2 3B Instruct Model
# This script uses a Modal Function to download the model into a cloud Volume.
#
# Run it with:
#    modal run download

import modal

MODELS_DIR = "/llamas"
DEFAULT_NAME = "meta-llama/Llama-3.2-3B-Instruct"

MINUTES = 60
HOURS = 60 * MINUTES

# Create a modal Volume to store the model
volume = modal.Volume.from_name("llamas", create_if_missing=True)

# This defines the image to use for the modal function
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "huggingface_hub",  # download models from the Hugging Face Hub
            "hf-transfer",  # download models faster with Rust
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# We run the function from a modal App, which will have our HF_SECRET env var set.
# Add your HuggingFace secret access token here: https://modal.com/secrets
# secret name: huggingface
# env var name: HF_TOKEN
app = modal.App(image=image, secrets=[modal.Secret.from_name("huggingface")])

# This function will be ran in the cloud, with the volume mounted.
@app.function(volumes={MODELS_DIR: volume}, timeout=4 * HOURS)
def download_model(model_name, force_download=False):
    from huggingface_hub import snapshot_download

    volume.reload()

    snapshot_download(
        model_name,
        local_dir=MODELS_DIR + "/" + model_name,
        ignore_patterns=[
            "*.pt",
            "*.bin",
            "*.pth",
            "original/*",
        ],  # Ensure safetensors
        force_download=force_download,
    )

    volume.commit()

    print("Model successfully downloaded")

@app.local_entrypoint()
def main(
    model_name: str = DEFAULT_NAME,
    force_download: bool = False,
):
    download_model.remote(model_name, force_download)
