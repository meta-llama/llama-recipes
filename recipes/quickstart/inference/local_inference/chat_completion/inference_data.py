import pandas as pd
import json

prompt_completions_path = "data.json"
output_path = "inference_data.json"

def main():
    df = pd.read_json(prompt_completions_path)
    df = df[-10:]

    def wrap_with_role(question: str) -> dict:
        return {"role": "user", "content": question}

    df["context"] = df["question"].apply(wrap_with_role)
    context_list = df["context"].apply(lambda x: [x]).tolist()

    with open(output_path, 'w') as f:
        json.dump(context_list, f, indent=4)

if __name__ == "__main__":
    main()
