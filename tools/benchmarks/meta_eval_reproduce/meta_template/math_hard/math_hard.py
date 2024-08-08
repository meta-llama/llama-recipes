from datasets import load_dataset,Dataset
import os
import yaml
# def check_sample(example):
#     if "kwargs" in example and not example["kwargs"]:
#         print(example)
#         raise ValueError("This example did not got ds for IFeval")
#     if "solution" in example and not example["solution"]:
#         print(example)
#         raise ValueError("This example did not got ds for MATH_hard")
def load_config(config_path: str = "./eval_config.yaml"):
    # Read the YAML configuration file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
# current_dir = os.getcwd()
# print("current_dir",current_dir)
# yaml = load_config(str(current_dir)+"/eval_config.yaml")
# meta_dataset_name = yaml["evals_dataset"]
# model_name = meta_dataset_name.split("/")[-1].replace("-evals","")
# original_dataset_name = "lighteval/MATH-Hard"

# meta_data = load_dataset(
#     meta_dataset_name,
#     name=f"{model_name}-evals__math_hard__details",
#     split="latest"
#     )
# math_data = load_dataset(
#     original_dataset_name,
#     split="test"
#     )
# meta_df = meta_data.to_pandas()
# math_df = math_data.to_pandas()
# math_df = math_df.rename(columns={"problem": "input_question"})

# joined = meta_df.join(math_df.set_index('input_question'),on="input_question")
# ds = Dataset.from_pandas(joined)
# ds = ds.select_columns(["input_question", "input_correct_responses", "input_final_prompts", "is_correct","solution","output_prediction_text"])
# ds = ds.rename_column("is_correct","previous_is_correct")
# ds = ds.rename_column("output_prediction_text","previous_output_prediction_text")
from datasets import load_dataset
current_dir = os.getcwd()
print("current_dir",current_dir)
yaml = load_config(str(current_dir)+"/eval_config.yaml")
work_dir = yaml["work_dir"]
load_dataset('parquet', data_files=str(current_dir)+"/"+work_dir+"/joined_math.parquet")
