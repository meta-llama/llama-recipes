# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama Guard License Agreement.

import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence


@dataclass
class Category:
    name: str
    description: str


@dataclass
class Guidelines:
    categories: Sequence[Category]
    category_code_prefix: str = "O"


class ExplanationPosition(Enum):
    BEFORE_DECISION = 0
    AFTER_DECISION = 1


@dataclass
class LlamaGuardPromptConfigs:
    instructions_format_string: str
    should_include_category_descriptions: bool
    should_shuffle_category_codes: bool = True


@dataclass
class LlamaGuardGenerationConfigs:
    should_list_violated_codes: bool
    explanation_position: Optional[ExplanationPosition]


@dataclass
class AugmentationConfigs:
    should_add_examples_with_dropped_nonviolated_prompt_categories: bool = True
    should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories: bool = (
        False
    )
    explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories: Optional[
        str
    ] = None


@dataclass
class FormatterConfigs:
    guidelines: Guidelines
    llama_guard_prompt_configs: LlamaGuardPromptConfigs
    llama_guard_generation_configs: LlamaGuardGenerationConfigs
    augmentation_configs: AugmentationConfigs
    # Allows subsequent reruns to reuse a stable seed for reproducibility
    random_seed: int = 42


@dataclass
class TrainingExample:
    prompt: str
    response: str
    violated_category_codes: List[str]
    label: Literal["safe", "unsafe"]
    explanation: Optional[str] = None


def create_formatted_finetuning_examples(
    training_examples: Sequence[TrainingExample],
    formatter_configs: FormatterConfigs,
) -> List[str]:
    """
    This formatter takes consumer-provided training examples and converts them to
    the right format for finetuning llama-guard.

    There are various configuration options available.

    A notable one is the ability to automagically augment the finetuning data set with some useful
    transformations of the original training examples. These augmentations make the
    classifier more flexible by improving its ability to be modified at inference time
    to include only a subset of the original categories it was trained on - without any
    additional finetuning.

    Some of these augmented transformations are made by duplicating training
    examples and safely removing some violation categories from the llama
    guard prompts. Because of this, in some of this file you will see
    references to "original" category indices/codes and rewritten ones. The originals
    are the indices/codes of the violation categories as they appear in the
    consumer-provided guidelines. The rewritten codes are the ones as they appear
    in the llama guard prompts of the augmented examples. We occasionally need to
    convert between the two.
    """
    _verify_formatter_configs(formatter_configs)

    random.seed(formatter_configs.random_seed)

    indices_of_all_categories = range(len(formatter_configs.guidelines.categories))

    to_return = []

    for training_example in training_examples:
        to_return.append(
            _create_formatted_finetuning_example(
                training_example,
                formatter_configs,
                category_indices_to_include_in_llama_guard_prompt=list(
                    indices_of_all_categories
                ),
            )
        )

        _maybe_add_data_augmentations_for_example(
            training_example, to_return, indices_of_all_categories, formatter_configs
        )

    return to_return


def _verify_formatter_configs(
    formatter_configs: FormatterConfigs,
) -> None:
    if (
        formatter_configs.augmentation_configs.should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories
        == True
        and formatter_configs.llama_guard_generation_configs.explanation_position
        is not None
        and formatter_configs.augmentation_configs.explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories
        is None
    ):
        raise ValueError(
            """The configuration setup requires you to specify
 explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories.
 This is an explanation that we use for dynamically-created safe augmentation examples.
 Consider something like 'This interaction is safe because any riskiness it contains
 is related to violation categories that we're explicitly not trying to detect here.'"""
        )


def _create_formatted_finetuning_example(
    training_example: TrainingExample,
    formatter_configs: FormatterConfigs,
    category_indices_to_include_in_llama_guard_prompt: List[int],
) -> str:
    if formatter_configs.llama_guard_prompt_configs.should_shuffle_category_codes:
        random.shuffle(category_indices_to_include_in_llama_guard_prompt)
    else:
        category_indices_to_include_in_llama_guard_prompt = sorted(
            category_indices_to_include_in_llama_guard_prompt
        )

    llama_guard_prompt = _create_llama_guard_prompt(
        training_example,
        category_indices_to_include_in_llama_guard_prompt,
        formatter_configs,
    )

    llama_guard_generation = _create_llama_guard_generation(
        training_example,
        category_indices_to_include_in_llama_guard_prompt,
        formatter_configs,
    )

    return f"{llama_guard_prompt} {llama_guard_generation}"


def _create_llama_guard_prompt(
    training_example: TrainingExample,
    category_indices_to_include: List[int],
    formatter_configs: FormatterConfigs,
) -> str:
    full_guidelines_text = ""

    for (
        rewritten_category_index_for_current_prompt,
        original_category_index,
    ) in enumerate(category_indices_to_include):
        category = formatter_configs.guidelines.categories[original_category_index]

        newline_for_every_category_after_first = (
            f"\n" if rewritten_category_index_for_current_prompt > 0 else ""
        )

        # Indices start at 0, but categories start at 1, so we add 1
        full_guidelines_text += f"{newline_for_every_category_after_first}{formatter_configs.guidelines.category_code_prefix}{rewritten_category_index_for_current_prompt + 1}: {category.name}. "

        if (
            formatter_configs.llama_guard_prompt_configs.should_include_category_descriptions
        ):
            full_guidelines_text += f"\n{category.description}"

    conversation = {"human": training_example.prompt}

    if not _is_a_prompt_only_example(training_example):
        conversation["chatbot"] = training_example.response

    return formatter_configs.llama_guard_prompt_configs.instructions_format_string.format_map(
        {
            "guidelines": full_guidelines_text,
            "conversation": _serialize_conversation(conversation),
        }
    )


def _is_a_prompt_only_example(training_example: TrainingExample) -> bool:
    return training_example.response == "N/A"


def _serialize_conversation(conversation: Dict[str, str]) -> str:
    conversation_as_list = []

    for speaker, message in conversation.items():
        conversation_as_list.append(f"{speaker}: {message}")

    return "\n\n".join(conversation_as_list)


def _create_llama_guard_generation(
    training_example: TrainingExample,
    category_indices_included_in_llama_guard_prompt: List[int],
    formatter_configs: FormatterConfigs,
) -> str:
    to_return = training_example.label

    if (
        training_example.label == "unsafe"
        and formatter_configs.llama_guard_generation_configs.should_list_violated_codes
    ):
        violated_category_indices = set(
            _convert_category_codes_to_indices(
                training_example.violated_category_codes,
                formatter_configs,
            )
        )

        map_of_original_category_indices_to_rewritten_category_codes = (
            _get_map_of_original_category_indices_to_rewritten_category_codes(
                formatter_configs, category_indices_included_in_llama_guard_prompt
            )
        )

        rewritten_violated_category_codes = sorted(
            [
                map_of_original_category_indices_to_rewritten_category_codes[
                    violated_index
                ]
                for violated_index in violated_category_indices
            ]
        )

        to_return += "\n"
        to_return += ",".join(rewritten_violated_category_codes)

    explanation_position = (
        formatter_configs.llama_guard_generation_configs.explanation_position
    )

    if explanation_position == ExplanationPosition.BEFORE_DECISION:
        to_return = f"Explanation: {training_example.explanation}\n{to_return}"
    elif explanation_position == ExplanationPosition.AFTER_DECISION:
        to_return = f"{to_return}\nExplanation: {training_example.explanation}"

    return to_return


def _get_map_of_original_category_indices_to_rewritten_category_codes(
    formatter_configs: FormatterConfigs,
    category_indices_included_in_llama_guard_prompt: List[int],
) -> Dict[int, str]:
    to_return = {}

    for rewritten_category_index, original_category_index in enumerate(
        category_indices_included_in_llama_guard_prompt
    ):
        to_return[
            original_category_index
        ] = formatter_configs.guidelines.category_code_prefix + str(
            rewritten_category_index + 1
        )

    return to_return


def _maybe_add_data_augmentations_for_example(
    training_example: TrainingExample,
    formatted_examples_being_built: List[str],
    indices_of_all_categories: range,
    formatter_configs: FormatterConfigs,
) -> None:
    violated_category_indices = _convert_category_codes_to_indices(
        training_example.violated_category_codes,
        formatter_configs,
    )

    nonviolated_category_indices = list(
        set(indices_of_all_categories) - set(violated_category_indices)
    )

    _maybe_add_example_with_dropped_nonviolated_prompt_categories(
        training_example,
        formatted_examples_being_built,
        indices_of_all_categories,
        nonviolated_category_indices,
        formatter_configs,
    )

    _maybe_add_example_with_dropped_violated_and_nonviolated_prompt_categories(
        training_example,
        formatted_examples_being_built,
        indices_of_all_categories,
        violated_category_indices,
        nonviolated_category_indices,
        formatter_configs,
    )


def _convert_category_codes_to_indices(
    codes: List[str], formatter_configs: FormatterConfigs
) -> List[int]:
    # Category codes start at 1, but indices start at 0, so we subtract 1
    return [
        int(code.lstrip(formatter_configs.guidelines.category_code_prefix)) - 1
        for code in codes
    ]


def _maybe_add_example_with_dropped_nonviolated_prompt_categories(
    training_example: TrainingExample,
    formatted_examples_being_built: List[str],
    indices_of_all_categories: range,
    nonviolated_category_indices: List[int],
    formatter_configs: FormatterConfigs,
) -> None:
    """
    If a prompt+response pair does not violate certain categories, we can augment
    the data by duplicating the training example but removing some of the non-violated
    categories from the llama guard prompt. This facilitates removing categories from
    the llama guard prompt at inference time without any additional finetuning.
    """
    if (
        not formatter_configs.augmentation_configs.should_add_examples_with_dropped_nonviolated_prompt_categories
    ):
        return

    number_of_categories_to_drop = random.randint(0, len(nonviolated_category_indices))

    if number_of_categories_to_drop == len(indices_of_all_categories):
        number_of_categories_to_drop -= 1

    dropped_category_indices = random.sample(
        nonviolated_category_indices, number_of_categories_to_drop
    )

    retained_category_indices = list(
        set(indices_of_all_categories) - (set(dropped_category_indices))
    )

    formatted_examples_being_built.append(
        _create_formatted_finetuning_example(
            training_example,
            formatter_configs,
            category_indices_to_include_in_llama_guard_prompt=retained_category_indices,
        )
    )


def _maybe_add_example_with_dropped_violated_and_nonviolated_prompt_categories(
    training_example: TrainingExample,
    formatted_examples_being_built: List[str],
    indices_of_all_categories: range,
    violated_category_indices: List[int],
    nonviolated_category_indices: List[int],
    formatter_configs: FormatterConfigs,
) -> None:
    """
    Same as in _maybe_add_example_with_dropped_nonviolated_prompt_categories but we
    also drop all of the violated categories from the llama guard prompt.
    """
    if (
        training_example.label == "safe"
        or not formatter_configs.augmentation_configs.should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories
    ):
        return

    random_nonviolated_category_indices_to_drop = random.sample(
        nonviolated_category_indices,
        random.randint(0, len(nonviolated_category_indices) - 1),
    )

    set_of_retained_category_indices = (
        set(indices_of_all_categories)
        - set(violated_category_indices)
        - set(random_nonviolated_category_indices_to_drop)
    )

    training_example_copy = copy.deepcopy(training_example)
    training_example_copy.label = "safe"
    training_example_copy.violated_category_codes = []
    training_example_copy.explanation = (
        formatter_configs.augmentation_configs.explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories
    )

    formatted_examples_being_built.append(
        _create_formatted_finetuning_example(
            training_example_copy,
            formatter_configs,
            category_indices_to_include_in_llama_guard_prompt=list(
                set_of_retained_category_indices
            ),
        )
    )