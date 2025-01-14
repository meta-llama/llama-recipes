# ## Generating HumanEval Results with our Llama 3.2 3B Instruct Model
# This app starts many parallel clients to send requests to the vLLM server.
# 
# For each of the tasks in the HumanEval test set, we'll run a client to request 1000 completions.
# Results are saved to our mounted volume.
#
# Run it with:
#    modal run generate --data-dir test --no-dry-run --n 1000 --subsample 100

from datetime import datetime
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import modal

# This defines the image to use for running openai clients in parallel
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "openai==1.38.0", "datasets==2.20.0"
)

app = modal.App("many-llamas-human-eval", image=image)

volume = modal.Volume.from_name("humaneval", create_if_missing=True)
DATA_DIR = Path("/mnt/humaneval")

default_system_prompt = "Write the body for the Python function provided in the prompt below. Do not write anything else. Your output will be directly concatenated with the prompt and the resulting function executed against tests."

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

@dataclass
class CompletionParams:
    model: str = None
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0
    presence_penalty: float = 0
    n: int = 1
    stop: str = None
    seed: int = None

@dataclass
class ClientParams:
    app_name: str = "many-llamas-human-eval"
    workspace: str = None
    api_key: str = "super-secret-token" # match the secret in inference.py

    @property
    def url(self):
        return f"https://{self.workspace}--{self.app_name}-serve.modal.run/v1"


@app.local_entrypoint()
def main(
    app_name: str = "many-llamas-human-eval",
    workspace: str = None,
    api_key: str = "super-secret-token",
    model: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    n: int = 1,
    stop: str = None,
    seed: int = None,
    data_dir: str = "dev-llm",
    subsample: int = 1, # percent of the test split to read
    system_prompt: str = default_system_prompt,
    dry_run: bool = True,
):
    if workspace is None:
        workspace = modal.config._profile

    client_params = ClientParams(app_name, workspace, api_key)

    completion_params = CompletionParams(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        n=n,
        stop=stop,
        seed=seed,
    )

    # Run a remote download function to save the HumanEval dataset in the cloud volume
    save_dataset.remote(path=data_dir, subsample=subsample)

    # Run a remote generation function
    results = run_human_eval.remote(
        client_params=client_params,
        completion_params=completion_params,
        system_prompt=system_prompt,
        data_dir=data_dir,
        dry_run=dry_run,
    )
    if results:
        with open("/tmp/results.jsonl", "w") as f:
            f.writelines(json.dumps(result) + "\n" for result in results)
        print(f"results saved locally to {f.name}")

# This is the parent function that spawns a client for each eval task
@app.function(volumes={DATA_DIR: volume}, timeout=1 * HOURS)
def run_human_eval(
    client_params: ClientParams,
    completion_params: CompletionParams,
    data_dir="dev-llm",
    system_prompt: str = default_system_prompt,
    dry_run=True,
):
    dataset = load_dataset(data_dir)

    timestamp = datetime.utcnow().isoformat() + "Z"
    output_dir = Path(DATA_DIR) / data_dir / f"run-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = []
    print(f"Eval set contains {len(dataset)} items")

    # For each eval item in the dataset, spawn a parallel openAI client worker that generates n completions each
    print(Colors.BOLD, f"Spawning clients for each eval item. You may notice a brief wait while the inference server(s) boot.", Colors.END, sep="")
    for i, item in enumerate(dataset):
        handles.append(
            run_item.spawn(
                item,
                client_params,
                completion_params,
                system_prompt,
                output_dir,
                dry_run,
            )
        )

    for handle in handles:
        result = handle.get()

    if not dry_run:
        return result

# This function is responsible for generating n completions for a single eval item
# It calls into our deployed vLLM server and saves results to the cloud volume
@app.function(volumes={DATA_DIR: volume}, timeout=1 * HOURS)
def run_item(
    item: dict,
    client_params: ClientParams,
    completion_params: CompletionParams,
    system_prompt: str,
    output_dir: Path,
    dry_run: bool,
):
    client = create_client(client_params)
    if not completion_params.model:
        model = client.models.list().data[0]
        model = model.id
        completion_params.model = model

    prompt = item["prompt"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    per_request = 250
    ct, completions = completion_params.n, []
    if not dry_run:
        while ct > 0:
            response = get_completion(
                client,
                messages=messages,
                **asdict(completion_params) | dict(n=min(ct, per_request)),
            )
            if response:
                completions += [
                    {
                        "task_id": item["task_id"],
                        "completion": choice.message.content,
                    }
                    for choice in response.choices
                ]
            ct -= per_request

        index = item["task_id"].split("/")[-1]
        output_path = output_dir / f"{index}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.writelines(json.dumps(completion) + "\n" for completion in completions)

        print(Colors.GREEN + f"Completions saved to {output_path}" + Colors.END)


class Colors:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"


def get_completion(client, **kwargs):
    try:
        response = client.chat.completions.create(**kwargs)
        return response
    except Exception as e:
        print(Colors.RED, f"Error during API call: {e}", Colors.END, sep="")
        return None


def create_client(client_params: ClientParams):
    from openai import OpenAI

    client = OpenAI(api_key=client_params.api_key)
    client.base_url = client_params.url

    return client

# This function downloads the HumanEval dataset
@app.function(volumes={DATA_DIR: volume})
def save_dataset(path="dev-llm", subsample: int = 1):
    import datasets

    path = DATA_DIR / path

    ds = datasets.load_dataset(
        "openai/openai_humaneval",
        # reads 0% to subsample% of the test split
        split=datasets.ReadInstruction("test", to=subsample, unit="%"),
    )

    ds.to_json(path / "data.jsonl")

    volume.commit()


def load_dataset(path="dev-llm"):
    import datasets

    path = DATA_DIR / path

    ds = datasets.load_dataset(path=str(path), data_files="data.jsonl")

    return ds["train"]
