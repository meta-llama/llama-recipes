from transformers import AutoModelForCausalLM, AutoTokenizer

# model_id = "HelixAI/codellama-8bit-json-24-02-08-mkt-research-v3-rerun_epoch_5"

# model = AutoModelForCausalLM.from_pretrained(model_id,force_download=True, resume_download=False)
# # print(model)


import pandas as pd
import datasets
import yaml

import pandas as pd
import datasets
import yaml
import random
import time
import torch
from peft import PeftModel

# tokenizer = AutoTokenizer.from_pretrained(
#     'codellama/CodeLlama-13b-Instruct-hf'
# )
device_arg = { 'device_map': 'auto' }
base_model = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-13b-Instruct-hf",
        return_dict=True,
        torch_dtype=torch.float16,
        **device_arg
    )

model = PeftModel.from_pretrained(base_model, "/home/ubuntu/llama-recipes-fork/llama-recipes/src/llama_recipes/models/codellama-8bit-json-mkt-research-24-03-04-merged_epoch_8", **device_arg)

tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/llama-recipes-fork/llama-recipes/src/llama_recipes/models/codellama-8bit-json-mkt-research-24-03-04-merged_tokenizer")
# okenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-13b-Instruct-hf")

# tokenizer.add_special_tokens(
#     {
#         "pad_token": "<PAD>"
#     }
# )
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "left"
# tokenizer.add_special_tokens(
#     {
#         "eos_token": "</s>",
#         "bos_token": "</s>",
#         "unk_token": "</s>",
#         "pad_token": '</s>',
#     }
# )

# tokenizer.pad_token = tokenizer.eos_token
def load_prompt_template(prompt_template_filename, user_text):
    with open(prompt_template_filename, 'r') as file:
        yaml_data = yaml.safe_load(file)
    prompt = yaml_data['prompt']
    return prompt.format(user_text=user_text)


start = time.time()
query = "What Is The performance For Buyout since Sept 30 2023"
prompt_template_filename = "/home/ubuntu/llama-recipes-fork/llama-recipes/src/llama_recipes/datasets/training_prompt_templates/hl_mr_prompt.yaml"
query_list = """Can you display funds in infrastructure that have upcoming deadlines
show me top asset managers in Secondaries by size in 2022
show me the biggest asset managers that invest in north american Buyout
which asset class had the most dry powder in 2020
What is the rate of contributions for Buyout funds?
provide me with a list of the top five asset managers in Real Estate
show me the biggest asset managers focused on Buyout in MENA
what are the various areas of strategy within private assets
What's doing better? buyout or vc?
what was the top performing focus in 2020(
What was the worst performing year for buyout in Europe - Western?
show me buyout funds trading in EUR
what was the top performing strategy in 2020
what are the top vintages for Venture Capital
show me the biggest asset managers that invest in emerging markets Co-Investment
what are best performing funds in Infrastructure?
Show me Secondaries funds that are fundraising with high management fees?
show me top Infrastructure funds
What Is The Median Performance On Clearlake Funds?
give funds with net_irr greater than 20 and netrvpi less than 1
show me top 5 vintages in Real Estate
show me top performing managers in Fund of Funds
What was the low performing asset class in 2008
show me top performing asset managers in Co-Investment
Which Mezzanine funds have the largest GP commitment?
what was the top performing sub asset class in 2020
what are returns of real estate sub asset class areas
show me low performing asset managers in Natural Resources
Show me all credit funds managed by Lightbay Capital
display the top 5 performing funds in the Secondaries category
Can you display the top five real estate funds for me
What is the typical performance of funds grouped by the decade in which they were established?
show me the biggest asset managers that invest in north american lending
provide me with a list of the top five asset managers in Natural Resources
show me top asset managers in lending by size in 2020
Show me the hamilton Lane worry index
who are the major players in debt in latin america
what was the average net dpi for infrastructure in 2022
what was the low performing asset class in 2019
who are the largest managers in asian Fund of Funds
How much money was raised for Infrastructure Funds last year?
which asset class had the most dry powder in 2010
What was the worst performing year for Real Estate in Latin America?
What are the largest Small Cap venture capital funds?
show me top 5 vintages in debt
what are the top 5 performing funds in Co-Investment
Can you display the top 5 funds with the best performance
what are typical returns of real estate sub asset class areas
show me the biggest asset managers focused on Buyout in Asia
who are the top Infrastructure managers
show me top asset managers in Credit by size in 2002
what are the different sub asset class areas within private assets
what is the best performing strategy in the private markets
Is distressed debt or origination raising more money?
show me top performing asset managers in Fund of Funds
Among these focus areas within Secondaries, which one is the largest?
show me the funds with the highest performance that strategy on technology investments
what investment styles exist in private equity underlying deals
who are the top Growth Equity managers
show me all funds trading in USD
In 2015, which asset class had the highest number of distributions
who are the major players in debt in asia?
Show me Venture Capital funds that are fundraising with low management fees?
show me top Growth Equity funds
What was the top performing asset class in 2022
Can you display the performance of venture capital investments from the 2010 vintage
what are best performing funds in Venture Capital
Median Irr Of hamilton Lane Funds
show me the biggest asset managers focused on Credit in MENA
What are the real estate funds currently in the process of raising funds
show me top 5 performing funds in buyout
Show historical contributions for private credit
Which funds are among the top 10 performers
What was the worst performing year for Infrastructure in North America?
which of these sub asset class areas within secondaries is the largest
show me manager with largest AUM focusing on Real Estate
What were distributions last year for private credit?
can you display private lending funds available for subscription
display the top 5 funds in debt with the best performance
managers with most AUM in Buyout in 2002?
what are the top 5 performing funds in Secondaries
show me top Co-Investment funds
show me the biggest asset managers focused on Infrastructure in Emerging Markets
show me top 5 asset managers in Secondaries
what are the top vintages for Real Estate
what investment styles exist within private equity
show me the venture capital fund with the highest return
Can you provide additional information about Novel coworking III-I, LP?
what are typical returns of real estate funds
Show me funds that are available for subscription right now?
show me the biggest asset managers that invest in global Real Estate
Which asset managers excel in the Buyout industry
show me top Buyout funds
who are the largest managers in asian Credit?
Display the best-performing funds that have exposure to North America
show me worst performing managers in Growth Equity
what are the top 10 best performing funds in Co-Investment
In what vintage years have growth funds outperformed PME?
show me top 5 vintages in Infrastructure
Show me Credit funds that are fundraising with terms less than 5 years?"""

query_list = query_list.split("\n")
# query_list = [
#     "Show me real estate funds that are fundraising and include their asset manager performance.",
#     "What is the status of Private Markets Fundraising?"
# ]
for query in query_list:
    base_prompt_template = load_prompt_template(prompt_template_filename, query)

    # JSON_BASE_PROMPT, prompt = apply_prompt_template()
    # base_prompt_template = JSON_BASE_PROMPT.format(prompt=prompt, user_text=query)

    B_INST, E_INST = "[INST]", "[/INST]"
    base_prompt_template = f"{tokenizer.bos_token}{B_INST} {base_prompt_template.strip()} {E_INST}"
    # prompt_tokens = [tokenizer.encode(, add_special_tokens=False)]

    # print("-"*100)
    # print("Input Prompt: ")
    # print(base_prompt_template)

    start = time.time()
    # print(base_prompt_template)
    # prompts=[base_prompt_template]
    # base_prompt_template=f"""[INST] <<SYS>>\n You are an AI assistant. Your job is to generate JSON to answer questions in the private markets space.\n  The JSON should include the python function to call and params to pass to the function. The functions execute the params and return dataframes.\n  The template of the JSON:\n  {{\n  "function": "<function_name>",\n  "params": {{\n      "filter_conditions": [],\n      "sort_conditions": [],\n      "aggregate_conditions": [],\n      "time": {{}}\n  }}\n  }}\n  Available functions:\n  - get_funds_by_status\n  - get_funds_with_upcoming_closes\n  - get_fund_info\n  - get_previous_vintages_for_fund\n  - get_funds_by_performance\n  - get_funds_by_outperformance\n  - get_managers_by_performance\n  - get_managers_by_size\n  - get_previous_vintages_for_family\n  - get_styles_by_performance\n  - get_focuses_by_performance\n  - get_styles_by_attribute\n  - get_focus_areas\n  - compare_fund_performance_by_size\n  - get_vintages_by_performance\n  - get_geographies_by_performance\n  - get_funds_by_status_manager_performance\n  - get_funds_by_status_family_performance\n  - get_managers_by_status_performance\n  \n  Columns in dataframes:\n  fund_name: str # Name of the fund\n  family_name: str # Name of the fund family\n  manager_name: str \n  style: str # Investment strategy or asset class\n  Unique style values possible: [\'Buyout\', \'Credit\', \'Infrastructure\', \'Real Estate\', \'Venture Capital\', \'Fund of Funds\', \'Secondaries\', \'Growth Equity\', \'Co-Investment\', \'Natural Resources\']\n  focus: str # Primary investment area or industry of focus\n  Unique focus values possible: [\'Small Cap\', \'Senior Debt\', \'Value Add\', \'Balanced\', \'Buyout\',\n     \'Distressed\', \'Mid Cap\', \'Seed/Early Stage\', \'Secondaries\',\n     \'Special Situations\', \'Opportunistic\', \'Venture Capital\',\n     \'Mezzanine\', \'Core\', \'Multi Manager\', \'Single Manager\',\n     \'Large Cap\', \'Late Stage\', \'Turnaround\', \'Real Estate\',\n     \'Multi Focus\', \'Fund Interests\', \'Energy\', \'Expansion Stage\',\n     \'Lending & Leasing\', \'Credit\', \'Direct Interests\', \'Agriculture\',\n     \'Growth Equity\', \'Timber\', \'Mining\', \'Royalty\', \'Infrastructure\',\n     \'Natural Resources\']\n  geo: str # Predominant regions or countries to invest\n  fund_status: str # Fundraising status of the fund\n  Unique values: [\'Out of Market\', \'Fundraising\', \'Projected\']\n  attribute: str\n  performance_metric: str\n  size_metric: str\n  formal_esg_policy: str # Whether the fund has a formal ESG policy Yes or No\n  ascending: bool\n  stat: str\n  start_time: str\n  end_time: str\n  start_year: int\n  end_year: int\nNote: You may only return JSON statements. For follow up questions that require information from the previous user message and ai response, use the relevant params from the previous JSON to include in the new one. It should only refer conversation history JSON when needed.\n<</SYS>>\nConversation history:\n\nHuman: {query}.\' [/INST]"""
    # print("base-prompt",base_prompt_template)
    # model_input = tokenizer.batch_encode_plus([base_prompt_template], 
    #                                               return_tensors="pt",
    #                                               padding=True,
    #                                               add_special_tokens=False)
    model_input = tokenizer.batch_encode_plus([base_prompt_template], return_tensors="pt", add_special_tokens=False) # padding="max_length", max_length=1500, truncation=True,  


    # model_input
    # token_num = model_input["input_ids"].size(-1)
    model_input["input_ids"] = model_input["input_ids"].to(model.device)
    sequence = model.generate(**model_input, max_new_tokens=256) # temperature=0.01, 

    predict = map(lambda x: x, tokenizer.batch_decode(sequence[:], skip_special_tokens=True))
    # print("total time ---", start-time.time())
    predictions = list(predict)[0]
    # print("-"*100)
    # print("AI Response: ")
    print(query,'@@@',predictions.split('/INST]')[1])
    # print("-"*100)
    # break
    
