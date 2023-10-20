import datasets
from llama_recipes.datasets.utils import Concatenator

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("kiamesdavies/prometheus-grafana-dashboards-full-v2",
                                    split="test" if split == "validation" else split)

    prompt = (
        f"{B_INST} {B_SYS}Provided series of PromQL queries for several Grafana dashboard panels, generative a full Grafana dashboard.{E_SYS}```json\n{{designer_input}}\n```\n{E_INST}\n```json\n{{designer_output}}\n```{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                designer_input=sample["designer_input"],
                designer_output=sample["designer_output"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.filter(lambda x: x["status"] == "active" and type(x["id"]) is int and x['id'] != 18888).map(
        apply_prompt_template, remove_columns=list(dataset.features))
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset
