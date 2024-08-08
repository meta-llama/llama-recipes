from datasets import load_dataset,Dataset

def get_ifeval_data(model_name,output_dir):
    if model_name not in ["Meta-Llama-3.1-8B-Instruct","Meta-Llama-3.1-70B-Instruct","Meta-Llama-3.1-405B-Instruct"]:
        raise ValueError("Only Meta-Llama-3.1-8B-Instruct, Meta-Llama-3.1-70B-Instruct, Meta-Llama-3.1-405B-Instruct models are supported for IFEval")
    original_dataset_name = "wis-k/instruction-following-eval"
    #meta_dataset_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-evals"
    meta_dataset_name = f"meta-llama/{model_name}-evals"
    meta_data = load_dataset(
        meta_dataset_name,
        name=f"{model_name}-evals__ifeval__strict__details",
        split="latest"
        )
    ifeval_data = load_dataset(
        original_dataset_name,
        split="train"
        )
    meta_data = meta_data.map(get_question)
    meta_df = meta_data.to_pandas()
    ifeval_df = ifeval_data.to_pandas()
    ifeval_df = ifeval_df.rename(columns={"prompt": "input_question"})
    print("meta_df",meta_df.columns)
    print(meta_df)
    print("ifeval_df",ifeval_df.columns)

    print(ifeval_df)

    joined = meta_df.join(ifeval_df.set_index('input_question'),on="input_question")
    joined = joined.rename(columns={"input_final_prompts": "prompt"})
    joined = joined.rename(columns={"is_correct": "previous_is_correct"})
    joined = Dataset.from_pandas(joined)
    joined = joined.select_columns(["input_question", "prompt", "previous_is_correct","instruction_id_list","kwargs","output_prediction_text","key"])
    joined.rename_column("output_prediction_text","previous_output_prediction_text")
    print(joined)
    for item in joined:
        check_sample(item)
    joined.to_parquet(output_dir + f"/joined_ifeval.parquet")
def get_math_data(model_name,output_dir):
    if model_name not in ["Meta-Llama-3.1-8B-Instruct","Meta-Llama-3.1-70B-Instruct","Meta-Llama-3.1-405B-Instruct"]:
        raise ValueError("Only Meta-Llama-3.1-8B-Instruct, Meta-Llama-3.1-70B-Instruct, Meta-Llama-3.1-405B-Instruct models are supported for MATH_hard")
    original_dataset_name = "lighteval/MATH-Hard"
    meta_dataset_name = f"meta-llama/{model_name}-evals"
    meta_data = load_dataset(
        meta_dataset_name,
        name=f"{model_name}-evals__math_hard__details",
        split="latest"
        )
    math_data = load_dataset(
        original_dataset_name,
        split="test"
        )
    meta_df = meta_data.to_pandas()
    math_df = math_data.to_pandas()
    math_df = math_df.rename(columns={"problem": "input_question"})
    print("meta_df",meta_df.columns)
    print(meta_df)
    print("math_df",math_df.columns)

    print(math_df)

    joined = meta_df.join(math_df.set_index('input_question'),on="input_question")
    # joined = Dataset.from_pandas(joined)
    # joined = joined.select_columns(["input_question", "input_correct_responses", "input_final_prompts", "is_correct","solution","output_prediction_text"])
    # joined = joined.rename_column("is_correct","previous_is_correct")
    # joined = joined.rename_column("output_prediction_text","previous_output_prediction_text")
    print(joined)
    # for item in joined:
    #     check_sample(item)
    joined.to_parquet(output_dir + f"/joined_math.parquet")
    #joined.save_to_disk(output_dir + f"/joined_math")
def get_question(example):
    try:
        example["input_question"] = eval(example["input_question"].replace("null","None").replace("true","True").replace("false","False"))["dialog"][0]["body"].replace("Is it True that the first song","Is it true that the first song").replace("Is the following True","Is the following true")
        example["input_final_prompts"] = example["input_final_prompts"][0]
        return example
    except:
        print(example["input_question"])
        return
def check_sample(example):
    if "kwargs" in example and not example["kwargs"]:
        print(example)
        raise ValueError("This example did not got joined for IFeval")
    if "solution" in example and not example["solution"]:
        print(example)
        raise ValueError("This example did not got joined for MATH_hard")
