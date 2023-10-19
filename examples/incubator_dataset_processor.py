import datasets
from llama_recipes.datasets.utils import Concatenator

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("kiamesdavies/prometheus-grafana-dashboards-full", split="test" if split == "validation" else split)

    prompt = (
        f"{B_INST} {B_SYS}Using the supplied Grafana dashboard graphs/panels in JSON – encompassing title, type, description, and associated metrics – and optionally the header of its associated group and a general dashboard summary, extract insights for each graph in clear, conversational language. These insights can later be transformed into queries. Make sure to point out any distinct groupings, filters, functions, or conditions that might be relevant.{E_SYS}```json\n{{incubate_input}}\n```\n{E_INST}\n```json\n{{incubate_output}}\n```{{eos_token}}"
    )
    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                incubate_input=sample["incubate_input"],
                incubate_output=sample["incubate_output"],
                eos_token=tokenizer.eos_token,
            )
        }
    dataset = dataset.filter(lambda x: x["status"] == "active" and type(x["id"]) is int and x['id'] != 18888).map(apply_prompt_template, remove_columns=list(dataset.features))
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset