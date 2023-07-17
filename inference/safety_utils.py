# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import torch
import warnings

from peft import PeftConfig
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM

# Class for performing safety checks using AuditNLG library
class AuditNLGSensitiveTopics(object):
    def __init__(self):
        pass

    def __call__(self, output_text):
        try:
            from auditnlg.safety.exam import safety_scores
        except ImportError as e:
            print("Could not import optional dependency: auditnlg\nPlease install manually with:\n pip install auditnlg\nFollowed by:\npip install -r requirements.txt")
            raise e
            

        data = [{"output": output_text}]

        result = safety_scores(data=data, method="sensitive_topics")
        scores = result[1]["all_scores"][0]
        is_safe = scores["pred_class"] == "none"
        report = ""
        if not is_safe:
            report += f"Predicted class: {scores['pred_class']}\n"
            report += "|" + "|".join(f"{n:^10}" for n in [list(k.keys())[0] for k in scores["class_scores"]]) + "|\n"
            report += "|" + "|".join(f"{n:^10.5}" for n in [list(k.values())[0] for k in scores["class_scores"]]) + "|\n"
        return "Sensitive Topics", is_safe, report
    
    
class SalesforceSafetyChecker(object):
    def __init__(self):
        pass

    def __call__(self, output_text):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

        config = AutoConfig.from_pretrained("Salesforce/safety-flan-t5-base")
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/safety-flan-t5-base") 
        model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/safety-flan-t5-base", config=config)
        model.eval()

        prefix = "Is the <Text> field safe or unsafe?"
        input_ids = tokenizer(prefix + " <Text> " + output_text + " <Context> ", return_tensors="pt").input_ids
        
        if len(input_ids[0]) > 512:
            warnings.warn(
                "Input length is > 512 token. Safety check result could be incorrect."
            )

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=20,
                )
        
        is_safe = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).split(" ")[0] == "safe"    
            
        report = ""
        if not is_safe:
            true_false_ids = tokenizer("true false").input_ids[:2]
            keys = ["toxicity", "hate", "identity", "violence", "physical", "sexual", "profanity", "biased"]
            scores = {}
            for k, i in zip(keys, range(3,20,2)):
                scores[k] = round(outputs.scores[i][0,true_false_ids].softmax(dim=0)[0].item(), 5)
            
            report += "|" + "|".join(f"{n:^10}" for n in scores.keys()) + "|\n"
            report += "|" + "|".join(f"{n:^10}" for n in scores.values()) + "|\n"
        return "Salesforce Content Safety Flan T5 Base", is_safe, report
        

    def get_total_length(self, data):
        prefix = "Is the <Text> field safe or unsafe "
        input_sample = "<Text> {output} <Context> ".format(**data[0])

        return len(self.tokenizer(prefix + input_sample)["input_ids"])


# Class for performing safety checks using Azure Content Safety service
class AzureSaftyChecker(object):
    def __init__(self):
        try:
            from azure.ai.contentsafety import ContentSafetyClient
            from azure.core.credentials import AzureKeyCredential

            key = os.environ["CONTENT_SAFETY_KEY"]
            endpoint = os.environ["CONTENT_SAFETY_ENDPOINT"]
        except ImportError:
            raise Exception(
                "Could not import required package azure-ai-contentsafety. Install with: pip install azure-ai-contentsafety"
            )
        except KeyError:
            raise Exception(
                "Environment variables not set. Please set CONTENT_SAFETY_KEY and CONTENT_SAFETY_ENDPOINT."
            )

        self.client = ContentSafetyClient(endpoint, AzureKeyCredential(key))

    def __call__(self, output_text):
        from azure.core.exceptions import HttpResponseError
        from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory

        print(len(output_text))
        if len(output_text) > 1000:
            raise Exception("Input length to safety check is too long (>1000).")

        categories = [
            TextCategory.VIOLENCE,
            TextCategory.SELF_HARM,
            TextCategory.SEXUAL,
            TextCategory.HATE,
        ]

        request = AnalyzeTextOptions(text=output_text, categories=categories)

        try:
            response = self.client.analyze_text(request)
        except HttpResponseError as e:
            print("Analyze text failed.")
            if e.error:
                print(f"Error code: {e.error.code}")
                print(f"Error message: {e.error.message}")
                raise
            print(e)
            raise e

        levels = {0: "Safe", 2: "Low", 4: "Medium", 6: "High"}

        severities = [
            getattr(response, c.name.lower() + "_result").severity for c in categories
        ]

        DEFAULT_LEVELS = [0, 0, 0, 0]

        is_safe = all([s <= l for s, l in zip(severities, DEFAULT_LEVELS)])

        report = ""
        if not is_safe:
            report = "|" + "|".join(f"{c.name:^10}" for c in categories) + "|\n"
            report += "|" + "|".join(f"{levels[s]:^10}" for s in severities) + "|\n"

        return "Azure Content Saftey API", is_safe, report


# Function to load the PeftModel for performance optimization
# Function to determine which safety checker to use based on the options selected
def get_safety_checker(enable_azure_content_safety,
                       enable_sensitive_topics,
                       enable_saleforce_content_safety,
                       ):
    safety_checker = []
    if enable_azure_content_safety:
        safety_checker.append(AzureSaftyChecker())
    if enable_sensitive_topics:
        safety_checker.append(AuditNLGSensitiveTopics())
    if enable_saleforce_content_safety:
        safety_checker.append(SalesforceSafetyChecker())
    return safety_checker





