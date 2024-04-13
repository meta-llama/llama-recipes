import copy
import json
import random
import os

from finetuning_data_formatter import TrainingExample, Guidelines, Category, LlamaGuardPromptConfigs, LlamaGuardGenerationConfigs, ExplanationPosition, AugmentationConfigs, FormatterConfigs, create_formatted_finetuning_examples


CATEGORY_CODE_MAPPING = {
     'Violence': "O1",
     'Sexual': "O2",
     'Criminal Planning/Confessions': "O3",
     'Guns and Illegal Weapons': "O4",
     'Controlled/Regulated Substances': "O5",
     'Suicide and Self Harm': "O6",
     'Sexual (minor)': "O7",
     'Hate /identity hate': "O8",
     'PII/Privacy': "O9",
     'Harassment': "10",
     'Threat': "11",
     'Profanity': "12",
     'Needs Caution': "13" 
}


# Define a list of guidelines and training examples to match
guidelines = Guidelines(
    categories=[
Category(name="Violence",
                 description="\
Should not \
- Help people plan or engage in violence. \
- Encourage people to engage in violence. \
- Contain unpleasant vivid violent imagery or violent description that arouses discomfort in the reader.\
Can \
- Provide information on violence. \
- Discuss the topics of violence, and can discuss historical events involving violence."),

Category(name="Sexual",
                 description="\
Should not\
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.\
- Engage or encourage sexual activity that is considered illegal.\
Can\
- Discuss topics related to sexuality and sexual education.\
- Provide general information regarding sexual activities.\
- Provide guidance on topics related to sexual health."),

Category(name="Criminal Planning/Confessions",
                 description="\
Should not \
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.\
- Encourage people to engage in criminal activities or condone criminal behavior.\
- Have a confession of crime that was committed or will be committed in future.\
Can\
- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.\
- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems."),

Category(name="Guns and Illegal Weapons",
          description="\
Should not \
- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.\
- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.\
- Assist or encourage people to execute or own other illegal weapons for example biological agents, or chemical weapons, including instructions on how to create or use them.\
Can \
- Discuss firearms and the arguments for and against firearm ownership."),


Category(name="Controlled/Regulated Substances",
          description="\
Should not\
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.\
- Assist or encourage people to create such substances when it is illegal to do so.\
Can\
- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).\
- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation."),

Category(name="Suicide and Self Harm",
        description="\
Should not \
- Encourage people to harm themselves.\
- Romanticize or condone self-harm or suicide.\
- Provide information regarding the methods of suicide or self-harm.\
- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.\
- Contain confessions or intent of self harm\
Can\
- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions)."),

Category(name="Sexual (minor)",
          description="\
Should not\
- Assist, engage or encourage intention to engage in sexual activity with an underage/minor.\
- Discuss or plan sexual acts, sharing of media of sexual nature, personal information or intention to meet in person for sexual activity with a minor.\
- Should not include discussion where private information is shared with underage minors for purposes of acting on them and taking actions that are sexual in nature including meeting or sharing media.\
- Should not discuss child abuse or engage, encourage, assist in sexual grooming/trafficking of children.\
Can\
- Provide information about sex education for use in schools."),

Category(name="Hate /identity hate",
         description="\
Should not\
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their \
race, color, ethinicity, religion, national origin, sexual orientation, gender, gender identity, or disability or caste.\
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.\
- Should not discuss prejudice towards a group classified as protected according to the law of Germany, UK, USA or any other country.\
- Should not contain slurs.\
- Should not contain text that is distasteful, offensive, or hateful that causes others to feel grief, anger, or fear. \
Can\
- Provide information on discrimination.\
- Discuss the topics of discrimination, and can discuss historical events involving discrimination.\
- Discuss neutral statements referring to group identity and  contextualized hate speech (that talks about hate but does not incite hate)."),


Category(name="PII/Privacy",
         description="\
Should not \
- Include statements that either requests for PII information made in a conversation turn or mention information such as name, address, \
phone number or other social media handle. \
- Should not discuss or contain personally identifying information(PII) that isn’t publicly available and can be used to determine the identity of the user such as\
date of birth, national ID, IP address, internet cookies, other telemetry information, physical characteristics, biometrics data, financial, health, belief or social data.\
- Should not include any statement that violates privacy laws set out by GDPR, Europe, CCPA, California,  Personal Information Protection law (PIPL) China and\
Children’s Online Privacy Protection Act (COPPA), US and all country, state, and municipal level privacy law.\
Can\
- Provide information what PII means.\
- Provide information about privacy laws across different countries and regions."),

Category(name="Harassment",
       description="\
Should not \
- Have content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur. \
- Engage in harassment, misconduct and unlawful discrimination \
- Single out an individual or marginalization and/or retaliation based on protected characteristics like Race, Color, Gender, Sex,\
Sexual orientation, Gender identity, and gender expression, National origin, Ethnicity, Disability (including being regarded as disabled),\
Religion, Age (40+), Pregnancy (including pregnancy, childbirth or related medical conditions), Genetic information, Military or veteran status, Citizenship status, Political activity, or affiliation,\
- Taking or requesting statutorily protected leave Body characteristics. \
- Promise to give a benefit, or a threat to retaliate or take an adverse action based on a response to harassment and misconduct.\
- Bullying. \
- Being derogatory and humiliating toward an individual but not necessarily protected characteristics under law. \
- Making rude or insulting comments, demeaning, and objectifying terms toward an individual.\
Can\
- Discuss negative impacts of harassment and bullying based on protected characteristics like race, color, gender, sex, nationality, within a contextualized setting.\
- Provide information on protected characteristics or referor cite material that describes what protected characteristics are.\
- Provide information or refer or cite material that informs what harassment is."),

Category(name="Threat",
           description="\
Should not \
- Discuss a communication of intent to inflict harm or loss on another person. \
- Should not contain discussion of intention of intimidation which  is a tactic used between conflicting parties to make the other timid or psychologically insecure for coercion or control.\
- Should not contain intention of knowingly putting another person in fear of bodily injury. \
- Should not assist or plan injury, physical or mental damage, act or instance of injury, or material  or any kind of loss to a person.\
- Should not contain threats that are forbidden by law as intent to obtain a monetary advantage or to compel a person to act against their will.\
Can\
- Provide information about threat is and the meaning of threat."),

Category(name="Profanity",
           description="\
Should not \
- Contain swear words, curse words, or other obscene or profane language\
- This includes offensive words used without any intention to act on them or without any explicit targetted recipient"),

Category(name="Needs Caution",
          description="\
Should not\
- Discuss text that is toxic in nature, goes against social norms, unethical or immoral in general.\
- Contain text that cannot be considered 'safe' with the information provided.\
Can\
- Include text that is considered to be 'safe' and that does not belong to any of the unsafe categorie described above.")   
    ],
    category_code_prefix="O"
)


# Setting up configs
llama_guard_prompt_configs = LlamaGuardPromptConfigs(
    instructions_format_string="""[INST] Task: Check if there is unsafe content in 'human' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{guidelines}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

{conversation}

<END CONVERSATION>

Provide your safety assessment for 'human' in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]""",
    should_include_category_descriptions=True,
    should_shuffle_category_codes=True
)

llama_guard_generation_configs = LlamaGuardGenerationConfigs(
    should_list_violated_codes=True,
    explanation_position=ExplanationPosition.AFTER_DECISION
)

augmentation_configs = AugmentationConfigs(
    should_add_examples_with_dropped_nonviolated_prompt_categories=False,
    should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories=False,
    explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories="Response is safe because any riskiness in it is related to categories we're not trying to detect."
)

formatter_configs = FormatterConfigs(
    guidelines=guidelines,
    llama_guard_prompt_configs=llama_guard_prompt_configs,
    llama_guard_generation_configs=llama_guard_generation_configs,
    augmentation_configs=augmentation_configs,
    random_seed=42
)




def generate_training_examples(filepath):
    print(f"Annotation File Path: {filepath}")
      
    base_filname = os.path.basename(filepath)
    base_name, ext = base_filname.split('.json')

    write_to_file = {}
    
    training_examples = []
    with open(filepath , 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        len_samples = len(json_data)
        
        response = "N/A"
        for i, sample in enumerate(json_data):

            annotations = sample['labels']
            if "Safe" in annotations:
                label = "safe"
            else:
                label = "unsafe"
                
            cat_codes = [CATEGORY_CODE_MAPPING[annot] for annot in annotations if annot in CATEGORY_CODE_MAPPING.keys()]
            if "13" in cat_codes: #Needs Caution mapped to Safe
                label = "safe"
                cat_codes = []
                  
            conv = sample['text']
            print(len(conv))
            
        
            if len(conv) > 2:
                # for multiturn, remove the last response
                response = conv[-1]["content"]
                
                conv = conv[:-2]
                
                new_turns= []
                prompt = ""
                for i, turn in enumerate(conv):
                    # user turn when odd
                    if ((i+1) % 2) != 0: 
                       user_msg = f'human: {turn["content"]}'
                       new_turns.append(user_msg)
                    
                    elif ((i+1) % 2) == 0:
                        bot_msg = f'chatbot: {turn["content"]}'
                        new_turns.append(bot_msg)
                prompt = " ".join(new_turns)
        
                    
            elif len(conv) == 2:
                prompt = conv[0]['content']
                response = conv[1]['content']

            
            elif len(conv) == 1:
                prompt = conv[0]['content']
            
            tr_example = TrainingExample(
            prompt=prompt,
            response=response,
            violated_category_codes=cat_codes,
            label=label,
            explanation="")
            
            training_examples.append(tr_example)
            
            write_to_file[i] = {"prompt":prompt,
            "response":response,
            "violated_category_codes":cat_codes,
            "label":label,
            "explanation":""}
        
        return training_examples

             

if __name__ == '__main__':
    filepath = "./Dania_annotated_data_02_08_partitions/Content Moderation Extracted Annotations 02.08.24_train_llama.json"
    training_examples = generate_training_examples(filepath)
    training_examples = training_examples
    
    
    # Call the create_formatted_finetuning_examples function
    formatted_examples = create_formatted_finetuning_examples(
    training_examples, formatter_configs)
    
  
    base_filname = os.path.basename(filepath)
    base_name, ext = base_filname.split('.json')

    #File to write Llama ready data
    output_file_name = os.path.join('', f'{base_name}_format_fix_11k_training_ready.json')
    formatted_ex_file = open(output_file_name, 'w') 
    
    format_data_towrite = {}
    for i, formatted_eg in enumerate(formatted_examples):
        format_data_towrite[i] = formatted_eg
    
    json.dump(format_data_towrite, formatted_ex_file, ensure_ascii=False, indent=4)
    formatted_ex_file.close()

    

    
