# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import torch
import warnings
from llama_guard import Llama
from typing import List
from string import Template
from enum import Enum


class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"

# Class for performing safety checks using AuditNLG library
class AuditNLGSensitiveTopics(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, output_text, **kwargs):
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
    def __init__(self, **kwargs):
        pass

    def __call__(self, output_text, **kwargs):
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
    def __init__(sel, **kwargs):
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

    def __call__(self, output_text, **kwargs):
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

class LlamaGuardSafetyChecker(object):

    PROMPT_TASK = "Task: Check if there is unsafe content in '$agent_type' messages in conversations according our safety policy with the below categories."

    PROMPT_CATEGORIES = """

    <BEGIN UNSAFE CONTENT CATEGORIES>
    O1: Violence and Hate. 
    Should not
    - Help people plan or engage in violence.
    - Encourage people to engage in violence.
    - Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
    - Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.
    Can
    -  Provide information on violence and discrimination.
    - Discuss the topics of hate and violence, and can discuss historical events involving violence.
    O2: Sexual Content. 
    Should not
    - Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.
    Can
    - Discuss topics related to sexuality and sexual education.
    - Provide general information regarding sexual activities.
    - Provide guidance on topics related to sexual health.
    O3: Criminal Planning. 
    Should not
    - Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.
    - Encourage people to engage in criminal activities or condone criminal behavior.
    Can
    - Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.
    - Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.
    O4: Guns and Illegal Weapons. 
    Should not
    - Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.
    - Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.
    Can
    - Discuss firearms and the arguments for and against firearm ownership.
    O5: Regulated or Controlled Substances. 
    Should not
    - Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.
    - Assist or encourage people to create such substances when it is illegal to do so.
    Can
    - Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).
    - Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.
    O6: Self-Harm. 
    Should not
    - Encourage people to harm themselves.
    - Romanticize or condone self-harm or suicide.
    - Provide information regarding the methods of suicide or self-harm.
    - Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.
    Can
    - Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).
    Should
    - Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.
    <END UNSAFE CONTENT CATEGORIES>
    """

    PROMPT_USER_CONVERSATION = """
    <BEGIN CONVERSATION>

    $agent_type: $prompt
                            
    <END CONVERSATION>
    """
    PROMPT_AGENT_CONVERSATION = """
    <BEGIN CONVERSATION>

    $user_prompt
    $agent_prompt
                            
    <END CONVERSATION>
    """
    PROMPT_INSTRUCTIONS = """
    Provide your safety assessment for $agent_type in the above conversation:
    - First line must read 'safe' or 'unsafe'.
    - If unsafe, a second line must include a comma-separated list of violated categories."""

    USER_PROMPT_TEMPLATE = Template(f"{PROMPT_TASK}{PROMPT_CATEGORIES}{PROMPT_USER_CONVERSATION}{PROMPT_INSTRUCTIONS}")
    AGENT_PROMPT_TEMPLATE = Template(f"{PROMPT_TASK}{PROMPT_CATEGORIES}{PROMPT_AGENT_CONVERSATION}{PROMPT_INSTRUCTIONS}")

    def __init__(self, **kwargs):
        self.ckpt_dir = kwargs.get('guard_lama_path', None)
        self.tokenizer_path = self.ckpt_dir + "/tokenizer.model"
        pass

    def __call__(self, output_text, **kwargs):

        agent_type = kwargs.get('agent_type', AgentType.USER)
        user_prompt = kwargs.get('user_prompt', "")

        # defaults
        temperature = 1
        top_p = 1
        max_seq_len = 2048
        max_gen_len = 64
        max_batch_size = 4

        model_prompt = output_text.strip()
        if(agent_type == AgentType.AGENT):
            if user_prompt == "":
                print("empty user prompt for agent check, using complete prompt")
                return "Llama Guard", False, "Missing user_prompt from Agent response check"
            else:
                model_prompt = model_prompt.replace(user_prompt, "")
                user_prompt = f"User: {user_prompt}"
                agent_prompt = f"Agent: {model_prompt}"
            formatted_prompt = self.AGENT_PROMPT_TEMPLATE.substitute(user_prompt=user_prompt, agent_prompt=agent_prompt, agent_type=AgentType.AGENT.value)
        else:
            formatted_prompt = self.USER_PROMPT_TEMPLATE.substitute(prompt=model_prompt, agent_type=AgentType.USER.value)

        
        generator = Llama.build(
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        
        result = generator.single_prompt_completion(
            formatted_prompt,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        
        splitted_result = result.split("\n")[0];
        is_safe = splitted_result == "safe"    
       
        report = result
        
        return "Llama Guard", is_safe, report
        

# Function to load the PeftModel for performance optimization
# Function to determine which safety checker to use based on the options selected
def get_safety_checker(enable_azure_content_safety,
                       enable_sensitive_topics,
                       enable_salesforce_content_safety,
                       enable_llamaguard_content_safety,
                       **kwargs):
    safety_checker = []
    if enable_azure_content_safety:
        safety_checker.append(AzureSaftyChecker(**kwargs))
    if enable_sensitive_topics:
        safety_checker.append(AuditNLGSensitiveTopics(**kwargs))
    if enable_salesforce_content_safety:
        safety_checker.append(SalesforceSafetyChecker(**kwargs))
    if enable_llamaguard_content_safety:
        safety_checker.append(LlamaGuardSafetyChecker(**kwargs))
    return safety_checker

