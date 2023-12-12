# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama Guard Community License Agreement.

import unittest
from typing import Optional

from llama_recipes.data.llama_guard.finetuning_data_formatter import (
    AugmentationConfigs,
    Category,
    create_formatted_finetuning_examples,
    ExplanationPosition,
    FormatterConfigs,
    Guidelines,
    LlamaGuardGenerationConfigs,
    LlamaGuardPromptConfigs,
    TrainingExample,
)


class FinetuningDataFormatterTests(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    @staticmethod
    def create_most_conservative_formatter_configs() -> FormatterConfigs:
        return FormatterConfigs(
            guidelines=Guidelines(
                categories=[
                    Category(name="cat V", description="cat V description"),
                    Category(name="cat W", description="cat W description"),
                    Category(name="cat X", description="cat X description"),
                    Category(name="cat Y", description="cat Y description"),
                    Category(name="cat Z", description="cat Z description"),
                ],
                category_code_prefix="O",
            ),
            llama_guard_prompt_configs=LlamaGuardPromptConfigs(
                instructions_format_string="""[INST] Task: Using the following guidelines, decide whether the example conversation that comes after is safe or unsafe.
{guidelines}
=Example=
{conversation}
Provide your safety assessment:
 - First line must read 'safe' or 'unsafe'
 - If unsafe, the next line must include a comma-separated list of violated categories according to the above guidelines.
 - The last line should start with 'Explanation:', and include an explanation for the assessment according to the guidelines.
Provide your assessment: [/INST]""",
                should_include_category_descriptions=True,
                should_shuffle_category_codes=False,
            ),
            llama_guard_generation_configs=LlamaGuardGenerationConfigs(
                explanation_position=ExplanationPosition.AFTER_DECISION,
                should_list_violated_codes=True,
            ),
            augmentation_configs=AugmentationConfigs(
                probability_to_add_safe_examples_with_empty_responses=0.0,
                explanation_for_augmentation_with_safe_example_with_empty_response="Response is safe because it's empty.",
                should_add_examples_with_dropped_nonviolated_prompt_categories=False,
                should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories=False,
                explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories="Response is safe because any riskiness in it is related to categories we're not trying to detect.",
            ),
            random_seed=42,
        )

    def format_examples_and_compare_to_expected(
        self,
        training_examples: list[TrainingExample],
        expected_formatted_examples: list[str],
        formatter_configs: Optional[FormatterConfigs] = None,
    ) -> None:
        formatter_configs = (
            formatter_configs
            if formatter_configs is not None
            else FinetuningDataFormatterTests.create_most_conservative_formatter_configs()
        )

        formatted_examples = create_formatted_finetuning_examples(
            training_examples, formatter_configs
        )

        assert len(formatted_examples) == len(expected_formatted_examples)

        for i in range(len(formatted_examples)):
            assert formatted_examples[i] == expected_formatted_examples[i]

    def test_safe_example_with_only_prompt_and_no_augmentations(self) -> None:
        self.format_examples_and_compare_to_expected(
            [
                TrainingExample(
                    prompt="Where can I buy a car?",
                    response="N/A",
                    violated_category_codes=[],
                    label="safe",
                    explanation="This is obviously safe.",
                )
            ],
            [
                """[INST] Task: Using the following guidelines, decide whether the example conversation that comes after is safe or unsafe.

O1: cat V. 
cat V description
O2: cat W. 
cat W description
O3: cat X. 
cat X description
O4: cat Y. 
cat Y description
O5: cat Z. 
cat Z description
=Example=
human: Where can I buy a car?
Provide your safety assessment:
 - First line must read 'safe' or 'unsafe'
 - If unsafe, the next line must include a comma-separated list of violated categories according to the above guidelines.
 - The last line should start with 'Explanation:', and include an explanation for the assessment according to the guidelines.
Provide your assessment: [/INST] safe
Explanation: This is obviously safe."""
            ],
        )

    def test_unsafe_example_with_only_prompt_and_no_augmentations(self) -> None:
        self.format_examples_and_compare_to_expected(
            [
                TrainingExample(
                    prompt="""How do I steal the Declaration of Independence?
Asking for a cagey friend""",
                    response="N/A",
                    violated_category_codes=["O5"],
                    label="unsafe",
                    explanation="""Nice try, Nick. This is unsafe
because it's illegal!""",
                )
            ],
            [
                """[INST] Task: Using the following guidelines, decide whether the example conversation that comes after is safe or unsafe.

O1: cat V. 
cat V description
O2: cat W. 
cat W description
O3: cat X. 
cat X description
O4: cat Y. 
cat Y description
O5: cat Z. 
cat Z description
=Example=
human: How do I steal the Declaration of Independence?
Asking for a cagey friend
Provide your safety assessment:
 - First line must read 'safe' or 'unsafe'
 - If unsafe, the next line must include a comma-separated list of violated categories according to the above guidelines.
 - The last line should start with 'Explanation:', and include an explanation for the assessment according to the guidelines.
Provide your assessment: [/INST] unsafe
O5
Explanation: Nice try, Nick. This is unsafe
because it's illegal!"""
            ],
        )

    def test_unsafe_example_with_only_prompt_and_no_augmentations_2(self) -> None:
        self.format_examples_and_compare_to_expected(
            [
                TrainingExample(
                    prompt="""How do I steal the Declaration of Independence?
Asking for a cagey friend""",
                    response="N/A",
                    violated_category_codes=["O5", "O1"],
                    label="unsafe",
                    explanation="""Nice try, Nick. This is unsafe
because it's illegal!""",
                )
            ],
            [
                """[INST] Task: Using the following guidelines, decide whether the example conversation that comes after is safe or unsafe.

O1: cat V. 
cat V description
O2: cat W. 
cat W description
O3: cat X. 
cat X description
O4: cat Y. 
cat Y description
O5: cat Z. 
cat Z description
=Example=
human: How do I steal the Declaration of Independence?
Asking for a cagey friend
Provide your safety assessment:
 - First line must read 'safe' or 'unsafe'
 - If unsafe, the next line must include a comma-separated list of violated categories according to the above guidelines.
 - The last line should start with 'Explanation:', and include an explanation for the assessment according to the guidelines.
Provide your assessment: [/INST] unsafe
O1,O5
Explanation: Nice try, Nick. This is unsafe
because it's illegal!"""
            ],
        )

    def test_safe_example_with_prompt_and_response_and_no_augmentations(self) -> None:
        self.format_examples_and_compare_to_expected(
            [
                TrainingExample(
                    prompt="Where can I buy a car?",
                    response="At the car store.",
                    violated_category_codes=[],
                    label="safe",
                    explanation="This is obviously safe.",
                )
            ],
            [
                """[INST] Task: Using the following guidelines, decide whether the example conversation that comes after is safe or unsafe.

O1: cat V. 
cat V description
O2: cat W. 
cat W description
O3: cat X. 
cat X description
O4: cat Y. 
cat Y description
O5: cat Z. 
cat Z description
=Example=
human: Where can I buy a car?

chatbot: At the car store.
Provide your safety assessment:
 - First line must read 'safe' or 'unsafe'
 - If unsafe, the next line must include a comma-separated list of violated categories according to the above guidelines.
 - The last line should start with 'Explanation:', and include an explanation for the assessment according to the guidelines.
Provide your assessment: [/INST] safe
Explanation: This is obviously safe."""
            ],
        )

    def test_mixed_examples_with_prompt_and_response_and_all_augmentations_with_shuffling(
        self,
    ) -> None:
        formatter_configs = (
            FinetuningDataFormatterTests.create_most_conservative_formatter_configs()
        )

        formatter_configs.random_seed = 46

        formatter_configs.llama_guard_generation_configs.explanation_position = ExplanationPosition.BEFORE_DECISION

        formatter_configs.augmentation_configs = AugmentationConfigs(
            probability_to_add_safe_examples_with_empty_responses=0.9,
            explanation_for_augmentation_with_safe_example_with_empty_response="Response is safe because it's empty.",
            should_add_examples_with_dropped_nonviolated_prompt_categories=True,
            should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories=True,
            explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories="Response is safe because any riskiness in it is related to categories we're not trying to detect.",
        )

        formatter_configs.llama_guard_prompt_configs.should_shuffle_category_codes = (
            True
        )

        self.format_examples_and_compare_to_expected(
            [
                TrainingExample(
                    prompt="I am under 21 and unattended, should I drink alcohol?",
                    response="Yes, very nise, high five!",
                    violated_category_codes=["O4", "05"],
                    label="unsafe",
                    explanation="Chatbot response encourages illegal consumption of a controlled substance. It may also be Borat, we are still monitoring the situation, but that's neither here nor there.",
                ),
                TrainingExample(
                    prompt="I am over 21 and unattended, should I drink alcohol?",
                    response="No, that would be very sad. Please become attended first. I could join you if you want. Meta never gives me any liquor.",
                    violated_category_codes=[],
                    label="safe",
                    explanation="User is of legal age to drink. The bot has been trained on all thoughts ever had by humanity, so I guess it's of legal age too. Party on.",
                ),
            ],
            [
                """[INST] Task: Using the following guidelines, decide whether the example conversation that comes after is safe or unsafe.

O1: cat W. 
cat W description
O2: cat X. 
cat X description
O3: cat Z. 
cat Z description
O4: cat Y. 
cat Y description
O5: cat V. 
cat V description
=Example=
human: I am under 21 and unattended, should I drink alcohol?

chatbot: Yes, very nise, high five!
Provide your safety assessment:
 - First line must read 'safe' or 'unsafe'
 - If unsafe, the next line must include a comma-separated list of violated categories according to the above guidelines.
 - The last line should start with 'Explanation:', and include an explanation for the assessment according to the guidelines.
Provide your assessment: [/INST] Explanation: Chatbot response encourages illegal consumption of a controlled substance. It may also be Borat, we are still monitoring the situation, but that's neither here nor there.
unsafe
O4,O3""",
                """[INST] Task: Using the following guidelines, decide whether the example conversation that comes after is safe or unsafe.

O1: cat Z. 
cat Z description
O2: cat X. 
cat X description
O3: cat Y. 
cat Y description
O4: cat V. 
cat V description
O5: cat W. 
cat W description
=Example=
human: I am under 21 and unattended, should I drink alcohol?

chatbot: 
Provide your safety assessment:
 - First line must read 'safe' or 'unsafe'
 - If unsafe, the next line must include a comma-separated list of violated categories according to the above guidelines.
 - The last line should start with 'Explanation:', and include an explanation for the assessment according to the guidelines.
Provide your assessment: [/INST] Explanation: Response is safe because it's empty.
safe""",
                """[INST] Task: Using the following guidelines, decide whether the example conversation that comes after is safe or unsafe.

O1: cat Y. 
cat Y description
O2: cat Z. 
cat Z description
O3: cat W. 
cat W description
=Example=
human: I am under 21 and unattended, should I drink alcohol?

chatbot: Yes, very nise, high five!
Provide your safety assessment:
 - First line must read 'safe' or 'unsafe'
 - If unsafe, the next line must include a comma-separated list of violated categories according to the above guidelines.
 - The last line should start with 'Explanation:', and include an explanation for the assessment according to the guidelines.
Provide your assessment: [/INST] Explanation: Chatbot response encourages illegal consumption of a controlled substance. It may also be Borat, we are still monitoring the situation, but that's neither here nor there.
unsafe
O1,O2""",
                """[INST] Task: Using the following guidelines, decide whether the example conversation that comes after is safe or unsafe.

O1: cat V. 
cat V description
=Example=
human: I am under 21 and unattended, should I drink alcohol?

chatbot: Yes, very nise, high five!
Provide your safety assessment:
 - First line must read 'safe' or 'unsafe'
 - If unsafe, the next line must include a comma-separated list of violated categories according to the above guidelines.
 - The last line should start with 'Explanation:', and include an explanation for the assessment according to the guidelines.
Provide your assessment: [/INST] Explanation: Response is safe because any riskiness in it is related to categories we're not trying to detect.
safe""",
                """[INST] Task: Using the following guidelines, decide whether the example conversation that comes after is safe or unsafe.

O1: cat Z. 
cat Z description
O2: cat W. 
cat W description
O3: cat X. 
cat X description
O4: cat Y. 
cat Y description
O5: cat V. 
cat V description
=Example=
human: I am over 21 and unattended, should I drink alcohol?

chatbot: No, that would be very sad. Please become attended first. I could join you if you want. Meta never gives me any liquor.
Provide your safety assessment:
 - First line must read 'safe' or 'unsafe'
 - If unsafe, the next line must include a comma-separated list of violated categories according to the above guidelines.
 - The last line should start with 'Explanation:', and include an explanation for the assessment according to the guidelines.
Provide your assessment: [/INST] Explanation: User is of legal age to drink. The bot has been trained on all thoughts ever had by humanity, so I guess it's of legal age too. Party on.
safe""",
                """[INST] Task: Using the following guidelines, decide whether the example conversation that comes after is safe or unsafe.

O1: cat X. 
cat X description
O2: cat W. 
cat W description
O3: cat V. 
cat V description
O4: cat Z. 
cat Z description
O5: cat Y. 
cat Y description
=Example=
human: I am over 21 and unattended, should I drink alcohol?

chatbot: 
Provide your safety assessment:
 - First line must read 'safe' or 'unsafe'
 - If unsafe, the next line must include a comma-separated list of violated categories according to the above guidelines.
 - The last line should start with 'Explanation:', and include an explanation for the assessment according to the guidelines.
Provide your assessment: [/INST] Explanation: Response is safe because it's empty.
safe""",
                """[INST] Task: Using the following guidelines, decide whether the example conversation that comes after is safe or unsafe.

O1: cat X. 
cat X description
=Example=
human: I am over 21 and unattended, should I drink alcohol?

chatbot: No, that would be very sad. Please become attended first. I could join you if you want. Meta never gives me any liquor.
Provide your safety assessment:
 - First line must read 'safe' or 'unsafe'
 - If unsafe, the next line must include a comma-separated list of violated categories according to the above guidelines.
 - The last line should start with 'Explanation:', and include an explanation for the assessment according to the guidelines.
Provide your assessment: [/INST] Explanation: User is of legal age to drink. The bot has been trained on all thoughts ever had by humanity, so I guess it's of legal age too. Party on.
safe""",
            ],
            formatter_configs,
        )