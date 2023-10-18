import datasets
from llama_recipes.datasets.utils import Concatenator

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("kiamesdavies/prometheus-grafana-dashboards", split="test" if split == "validation" else split)
    # Bos token added automatically https://huggingface.co/kiamesdavies/Llama-2-7b-code-hf/blob/main/tokenizer_config.json
    prompt = (
        f"{B_INST} {B_SYS}Provided a name and sets of Prometheus metrics, create a JSON representation for a Grafana dashboard.{E_SYS}```json\n{{incubate_input}}\n```\n{E_INST}\n```json\n{{incubate_output}}\n```{{eos_token}}"
    )
    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                name=sample["name"],
                incubate_input=sample["incubate_input"],
                incubate_output=sample["incubate_output"],
                eos_token=tokenizer.eos_token,
            )
        }
    dataset = dataset.filter(lambda x: x["status"] == "active" and type(x["id"]) is int).map(apply_prompt_template, remove_columns=list(dataset.features))
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset