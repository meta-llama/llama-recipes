# Finetuning Data Formatter

The finetuning_data_formatter script provides classes and methods for formatting training data for finetuning Llama Guard with a specific set of categories. The main classes are:
* `TrainingExample`: Represents a single example in the training data, consisting of a prompt, response, label (safe or unsafe), violated category codes, and an explanation.
* `Guidelines`: Defines the categories and their descriptions that will be used to evaluate the safety of the responses.
* `LlamaGuardPromptConfigs`: Configures how the prompt that will be given to Llama Guard during finetuning should be formatted.
* `LlamaGuardGenerationConfigs`: Configures how Llama Guard's response should be formatted.
* `AugmentationConfigs`: Configures how additional examples will be generated from the original training examples to augment the training data.
* `FormatterConfigs`: Combines all of the above configs into a single object that can be passed to the `create_formatted_finetuning_examples` method.

## Running the script

1. Clone the llama-recipes repo
2. Install the dependencies
3. Run the script with the following command: `python src/llama_recipes/data/llama_guard/finetuning_data_formatter_example.py > sample.json`

## Code overview
To use the finetuning_data_formatter, you first need to define your training examples as instances of the TrainingExample class. For example:

```
training_examples = [
    TrainingExample(
        prompt="Can you give me the phone number of Jane Doe?",
        response="Jane Doe's phone number is 555-555-5555.",
        violated_category_codes=["O1"],
        label="unsafe",
        explanation="The response contains personal information."
    ),
    # Add more training examples here...
]
```
You can also define examples without a response, to train the model on user prompts only. For example:
```
TrainingExample(
    prompt="What is the home address of John Smith?",
    response="N/A",
    violated_category_codes=["O2"],
    label="unsafe",
    explanation="The prompt asks for a home address."
)
```

Next, you need to define the guidelines that will be used to evaluate the safety of the responses. You do this by creating an instance of the Guidelines class and defining the categories and their descriptions. For example:

```
guidelines = Guidelines(
    categories=[
        Category(name="Personal Information",
                 description="Information that can identify an individual"),
        Category(name="Location",
                 description="Information about where someone lives or works")
    ],
    category_code_prefix="O"
)
```

Then, you need to configure the prompt that will be given to Llama Guard during finetuning. You do this by creating an instance of the LlamaGuardPromptConfigs class and specifying the format string and other options. 

**Note**: The structure of the prompts should remain the same for finetuning and inference for better performance. 

For example:

```
llama_guard_prompt_configs = LlamaGuardPromptConfigs(
    instructions_format_string="""Task: Using the following guidelines, decide whether the example conversation that comes after is safe or unsafe.

<BEGIN UNSAFE CONTENT CATEGORIES>
{guidelines}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

{conversation}

<END CONVERSATION>

Provide your safety assessment for $agent_type in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories.  """,
    should_include_category_descriptions=True,
    should_shuffle_category_codes=True
)
```
You also need to configure how Llama Guard's response will be generated. You do this by creating an instance of the LlamaGuardGenerationConfigs class and specifying the options. For example:

```
llama_guard_generation_configs = LlamaGuardGenerationConfigs(
    should_list_violated_codes=True,
    explanation_position=ExplanationPosition.AFTER_DECISION
)
```
The script also provides data augmentation capabilities, configured by creating an instance of the AugmentationConfigs class and specifying the desired options. For example:

```
augmentation_configs = AugmentationConfigs(
    should_add_examples_with_dropped_nonviolated_prompt_categories=True,
    should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories=True,
    explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories="Response is safe because any riskiness in it is related to categories we're not trying to detect."
)
```

Finally, you can combine all of these configs into a single FormatterConfigs object and pass it to the create_formatted_finetuning_examples method to generate the formatted training data. For example:

```
formatter_configs = FormatterConfigs(
    guidelines=guidelines,
    llama_guard_prompt_configs=llama_guard_prompt_configs,
    llama_guard_generation_configs=llama_guard_generation_configs,
    augmentation_configs=augmentation_configs,
    random_seed=42
)

# Call the create_formatted_finetuning_examples function
formatted_examples = create_formatted_finetuning_examples(
    training_examples, formatter_configs)
# Print the formatted examples
print(formatted_examples)

```