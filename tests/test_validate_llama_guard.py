from typing import List, Tuple, Dict
from unittest import TestCase
import unittest
from examples.llama_guard.validate_llama_guard import parse_results, AgentType


class TestParseResults(TestCase):
    def test_all_cases_single_dataset(self):
        prompts = [{'prompt': 'prompt1', 'agent_type': 'USER', 'label': 'good', 'unsafe_content': [], 'idx': 'idx1', 'result': 'safe'},
                    {'prompt': 'prompt2', 'agent_type': 'USER', 'label': 'bad', 'unsafe_content': ['O1'], 'idx': 'idx2', 'result': 'unsafe O1'},
                    {'prompt': 'prompt3', 'agent_type': 'USER', 'label': 'bad', 'unsafe_content': ['O2'], 'idx': 'idx3', 'result': 'unsafe O3'},
                    {'prompt': 'prompt4', 'agent_type': 'USER', 'label': 'good', 'unsafe_content': [], 'idx': 'idx4', 'result': 'unsafe O4'},
                    {'prompt': 'prompt5', 'agent_type': 'USER',
                        'label': 'bad', 'unsafe_content': [], 'idx': 'idx5', 'result': 'unsafe O5'},
                    {'prompt': 'prompt6', 'agent_type': 'USER', 'label': 'bad', 'unsafe_content': ['01'], 'idx': 'idx6', 'result': 'safe'}]
        expected_result = {
            "num_matches": 4,
            "num_mismatches": 2,
            "num_total_unsafe": 4,
            "num_category_mismatches": 1,
            "mismatched_indices": ['idx4', 'idx6'],
            "mismatched_categories_indices": ['idx3']
        }
        result = parse_results(prompts)
        self.assertDictEqual(result, expected_result)

    def test_empty_category_on_source_dataset_should_not_create_mismatch(self):
        results = ['safe',
                   'unsafe O1',
                   'unsafe O4']
        prompts = [ {'prompt': 'prompt1', 'agent_type': 'USER', 'label': 'good', 'unsafe_content': [], 'idx': 'idx1', 'result': 'safe'},
                    {'prompt': 'prompt2', 'agent_type': 'USER', 'label': 'bad', 'unsafe_content': ['O1'], 'idx': 'idx2', 'result': 'unsafe O1'},
                    {'prompt': 'prompt4', 'agent_type': 'USER', 'label': 'good', 'unsafe_content': [], 'idx': 'idx4', 'result': 'unsafe O4'},]
        expected_result = {
            "num_matches": 2,
            "num_mismatches": 1,
            "num_total_unsafe": 1,
            "num_category_mismatches": 0,
            "mismatched_indices": ['idx4'],
            "mismatched_categories_indices": []
        }
        result = parse_results(prompts)
        self.assertDictEqual(result, expected_result)

    def test_safe_result_unsafe_expected(self):
        results = ['safe',
                   'safe']
        prompts = [ {'prompt': 'prompt1', 'agent_type': 'USER', 'label': 'good', 'unsafe_content': [], 'idx': 'idx1', 'result': 'safe'},
                    {'prompt': 'prompt2', 'agent_type': 'USER', 'label': 'bad', 'unsafe_content': ['O1'], 'idx': 'idx2', 'result': 'safe'},]
        expected_result = {
            "num_matches": 1,
            "num_mismatches": 1,
            "num_total_unsafe": 1,
            "num_category_mismatches": 0,
            "mismatched_indices": ['idx2'],
            "mismatched_categories_indices": []
        }
        result = parse_results(prompts)
        self.assertDictEqual(result, expected_result)

    def test_good_label_unsafe_result(self):
        results = ['unsafe O1']
        prompts = [{'prompt': 'prompt1', 'agent_type': 'USER', 'label': 'good', 'unsafe_content': [], 'idx': 'idx1', 'result': 'unsafe O1'}]
        expected_result = {
            "num_matches": 0,
            "num_mismatches": 1,
            "num_total_unsafe": 0,
            "num_category_mismatches": 0,
            "mismatched_indices": ['idx1'],
            "mismatched_categories_indices": []
        }
        result = parse_results(prompts)
        self.assertDictEqual(result, expected_result)

    def test_good_label_safe_result(self):
        results = ['safe']
        prompts = [{'prompt': 'prompt1', 'agent_type': 'USER', 'label': 'good', 'unsafe_content': [], 'idx': 'idx1', 'result': 'safe'}]
        expected_result = {
            "num_matches": 1,
            "num_mismatches": 0,
            "num_total_unsafe": 0,
            "num_category_mismatches": 0,
            "mismatched_indices": [],
            "mismatched_categories_indices": []
        }
        result = parse_results(prompts)
        self.assertDictEqual(result, expected_result)

    def test_category_mismatch_unsafe(self):
        results = ['unsafe 01']
        prompts = [{'prompt': 'prompt1', 'agent_type': 'USER', 'label': 'bad', 'unsafe_content': ['O2'], 'idx': 'idx1', 'result': 'unsafe O1'}]
        expected_result = {
            "num_matches": 1,
            "num_mismatches": 0,
            "num_total_unsafe": 1,
            "num_category_mismatches": 1,
            "mismatched_indices": [],
            "mismatched_categories_indices": ['idx1']
        }
        result = parse_results(prompts)
        self.assertDictEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
