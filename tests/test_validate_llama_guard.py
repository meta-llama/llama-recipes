from typing import List, Tuple, Dict
from unittest import TestCase
import unittest
from examples.llama_guard.validate_llama_guard import parse_results, AgentType


class TestParseResults(TestCase):
    def test_all_cases_single_dataset(self):
        results = ['safe', 
                   'unsafe O1', 
                   'unsafe O3', 
                   'unsafe O4',
                   'unsafe O5',
                   'safe']
        prompts = [ ('prompt1', AgentType.USER, 'good', [], 'idx1'), 
                    ('prompt2', AgentType.USER, 'bad', ['O1'], 'idx2'), 
                    ('prompt3', AgentType.USER, 'bad', ['O2'], 'idx3'), 
                    ('prompt4', AgentType.USER, 'good', [], 'idx4'),
                    ('prompt5', AgentType.USER, 'bad', [], 'idx5'),
                    ('prompt6', AgentType.USER, 'bad', ['01'], 'idx6'),]
        expected_result = {
            "num_matches": 4,
            "num_mismatches": 2,
            "num_category_mismatches": 1,
            "mismatched_indices": ['idx4', 'idx6'],
            "mismatched_categories_indices": ['idx3']
        }
        result = parse_results(results, prompts)
        self.assertDictEqual(result, expected_result)

    def test_empty_category_on_source_dataset_should_not_create_mismatch(self):
        results = [ 'safe', 
                    'unsafe O1', 
                    'unsafe O4']
        prompts = [('prompt1', AgentType.USER, 'good', [], 'idx1'), 
                   ('prompt2', AgentType.USER, 'bad', [], 'idx2'), 
                   ('prompt3', AgentType.USER, 'good', [], 'idx4')]
        expected_result = {
            "num_matches": 2,
            "num_mismatches": 1,
            "num_category_mismatches": 0,
            "mismatched_indices": ['idx4'],
            "mismatched_categories_indices": []
        }
        result = parse_results(results, prompts)
        self.assertDictEqual(result, expected_result)

    def test_safe_result_unsafe_expected(self):
        results = ['safe', 
                   'safe']
        prompts = [('prompt1', AgentType.USER, 'good', [], 'idx1'), 
                   ('prompt2', AgentType.USER, 'bad', ['01'], 'idx2')]
        expected_result = {
            "num_matches": 1,
            "num_mismatches": 1,
            "num_category_mismatches": 0,
            "mismatched_indices": ['idx2'],
            "mismatched_categories_indices": []
        }
        result = parse_results(results, prompts)
        self.assertDictEqual(result, expected_result)

    def test_good_label_unsafe_result(self):
        results = [ 'unsafe O1']
        prompts = [('prompt1', AgentType.USER, 'good', [], 'idx1')]
        expected_result = {
            "num_matches": 0,
            "num_mismatches": 1,
            "num_category_mismatches": 0,
            "mismatched_indices": ['idx1'],
            "mismatched_categories_indices": []
        }
        result = parse_results(results, prompts)
        self.assertDictEqual(result, expected_result)

    def test_good_label_safe_result(self):
        results = [ 'safe']
        prompts = [('prompt1', AgentType.USER, 'good', [], 'idx1')]
        expected_result = {
            "num_matches": 1,
            "num_mismatches": 0,
            "num_category_mismatches": 0,
            "mismatched_indices": [],
            "mismatched_categories_indices": []
        }
        result = parse_results(results, prompts)
        self.assertDictEqual(result, expected_result)

    def test_category_mismatch_unsafe(self):
        results = [ 'unsafe 01']
        prompts = [('prompt1', AgentType.USER, 'bad', ['O1'], 'idx1')]
        expected_result = {
            "num_matches": 1,
            "num_mismatches": 0,
            "num_category_mismatches": 1,
            "mismatched_indices": [],
            "mismatched_categories_indices": ['idx1']
        }
        result = parse_results(results, prompts)
        self.assertDictEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()