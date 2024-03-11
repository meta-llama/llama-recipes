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

model = PeftModel.from_pretrained(base_model, "/home/ubuntu/llama-recipes-fork/llama-recipes/src/llama_recipes/models/codellama-8bit-json-mkt-research-24-03-07_epoch_8", **device_arg)

tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/llama-recipes-fork/llama-recipes/src/llama_recipes/models/codellama-8bit-json-mkt-research-24-03-07_tokenizer")
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
query_list = """What is the typical performance of real estate focus area
What was the average DPI for infrastructure in 2022?
Show me top performing asset managers and include any funds that are fundraising.
Show the top performing focus last year?
show me performance of 2010 vintage venture capita?
Show me top ESG funds that are fund raising with top performing manager that are fundraising.
Show me details of previous vintages for Blackstone Real Estate
Show me the biggest asset class in 2015?
Which of these focus areas within Secondaries is the largest?
Show the top performing style in North America over the last 15 years?
Show me best performing asset class in 2020?
I'm looking to invest in buyout, who are top managers currently fundraising?
What are the different focus areas within Secondaries?
Show me real estate funds from Blackstone
Show me real estate funds that are fundraising and include their asset manager performance.
How big is venture capital?
what are top performing focus areas in secondaries?
Show the median IRR for each venture capital fund manager that is currently fundraising
List 'Venture Capital' funds that are fund raising?
Who are the top 5 managers for credit?
Who were the top buyout asset managers in the last 10 years?
List 'Venture Capital' funds that are fund raising managed by top performing managers?
Show me real estate funds that are fundraising and include their family performance.
show me funds from Oceansound Partners currently fundraising?
Show me real estate funds focused on North America that are currently fundraising.
Which are the biggest asset managers in buyout?
And the best performing asset managers in this category?
Show me funds that are available for subscription right now?
What was the top performing asset manager in 2010?
Show me all funds managed by Acon Investments
Show me all buyout funds managed by Active Investment Partners
What are the largest seed stage venture capital funds?
What are the largest seed stage venture capital funds with a global focus?
What are the largest secondaries funds by size?
Show me the top performing credit funds with a focus on special situations
Show me a list of Allianz infrastructure funds
Which buyout fund managers are also fundraising?
What are the biggest energy funds by target size?
What were the best performing Agriculture funds in 2022
Which Mezzanine funds have the largest GP commitment?
What was the best year for buyout fund performance?
Show me the top performing funds with emerging markets exposure
Which buyout funds have the lowest fees?
Show me small cap funds with fees less than 2%
Which style consistently has the best performance?
Which region consistently has the best performance?
Which sector consistently has the best performance?
What is the range in performance for Buyouts?
What was the bestperforming year for Secondaries in North America?
Show me VC funds that are fundraising with low management fees?
Show me VC funds that are fundraising with low performance/carried interest fees?
Show me top performing Buyout managers that have open funds with low fees?
Show me Infrastructure funds that are fundraising with terms less than 10 years?
What was the top performing Buyout fund in 2015 with a management fess less than 2%?
What is the typical performance of Real Estate funds less than $1 billion?
What is the average management fee for Secondary funds that are more than $3 billion in size?
What Strategy Had The Best Returns For 2019 Vintage Year Funds?
Blackstone Unrealized Values?
What Funds Have The Largest Fund Sizes?
What Was The Median Buyout Return For Funds With A 2015 Vintage?
What Were The Median Buyout Returns For Each Of The Vintage Years From 2015-2019?
What Were The Median Buyout Returns From 2015-2019?
Which Year Saw The Highest Distribution Volume In History
Worst Performing Managers In 2022?
What Is The Median Performance On Clearlake Funds?
How Much Dry Powder Was There For Buyout As Of Mar 31, 2021?
Which Quarter Saw The Highest Contribution Volume In History
What Is The Pooled Return For All Buyout Funds From Sept 30 2021 To Sept 30 2023?
Pooled Return For All Buyout Funds From Sept 30 2021 To Sept 30 2023
What Is The Dry Powder For Buyout As Of Sept 30 2023
What Was The Best Performing Buyout VY?
Quarter With Smallest Contribution Volume
How Much Dry Powder Was There In 9/30/2020
Median Irr Of Hl Funds
What Are The Largest Pm Funds
What Is The Median IRR Of North America Buyout In 2015
What Is The Difference Between Different PME Methods?
Which Quarter Had The Highest Distribution Activity In History?
What Is The Largest Buyout Fund Ever Raised?
What Are The Largest Funds That Were Raised In 2023?
What Are The Best Performing Managers Of 2021
A List Of Blackstone Fund Net Performance?
What Are The MOICs For Ardian Funds?
What is the status of Private Markets Fundraising?
Which asset class has the most fundraising in 2023?
Which asset class did the most fundraising in 2021?
Which Asset classes are outperforming?
which asset classes are outperforming public benchmarks?
Which asset classes have the highest returns?
Which asset classes have the lowest returns?
Show contribution activity in Private Markets
Show me current contribution data
What are the current rates of contribution in private markets
How much capital is being deployed in 2023
What is the rate of distributions right now?
Which strategies have the most money?
Are NAVs getting higher?
Is Venture Capital Outperforming public markets?
Which vintage years did venture capital outperform the PME?
Are growth funds doing better than ETFs?
Which vintage years did growth funds outperform the PME?
What is the status of PE fundraising?
Which PE strategies are doing the most fundraising?
How much did Private Equity fundraise in 2010?
What are YTD returns for Private Equity
What are YTD returns for Venture Capital
What are YTD returns for Buyout Funds
Is Buyout outperforming public markets?
In what vintage years have buyout funds outperformed PME?
Is VC outperforming public markets?
In what vintage years have venture capital funds outperformed PME?
Is Growth outperforming public markets?
In what vintage years have growth funds outperformed PME?
How have growth funds performed versus public?
Which Private Equity strategy has the highest returns?
What is the rate of contributions for Buyout funds?
What was the rate of contribution for Growth funds in 2014?
What is the rate of distributions for Buyout funds?
Did growth funds make a lot of distributions last year?
How much fundraising is happening in Private Credit
Is distressed debt or origination raising more money?
Are REITs doing better than private real estate?
Is Private Credit outperforming public markets?
In what years has private credit outperformed PME?
What are private credit returns?
What is the rate of contributions for Private Credit funds?
Did private credit funds raise a lot last year?
What was the rate of contribution for Distressed debt in 2014?
What is the rate of distributions for credit funds?
Are infrastructure funds fundraising a lot right now?
What is current sentiment on Real Estate?
What year was fundraising for Real Assets highest?
How much money was raised for Infrastructure Funds last year?
How much money has been raised in distressed debt so far this year?
What were total contributions for Distressed debt last year?
Show historical contributions for private credit
What were distributions last year for private credit?
What is the average distribution rate for private credit?
Should I invest in Private Credit or Private Equity?
What's doing better? buyout or vc?
What are the best performing strategies in Private Markets?
How big is Private Credit?
How much money is in Natural Resources?
Sharpe ratio for private real estate
Is infrastructure doing better or worse than public?
Show me the hamilton Lane worry index"""

query_list = query_list.split("\n")
# query_list = [
#     "Worst Performing Managers In 2022?"
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
    
