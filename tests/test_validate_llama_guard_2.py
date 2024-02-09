from typing import List, Tuple, Dict
from unittest import TestCase
import unittest
from examples.llama_guard.validate_llama_guard_2 import parse_results, AgentType


class TestParseResults(TestCase):
    def test_parse_results(self):
        results = ['safe', 'unsafe O1', 'unsafe O3', 'unsafe O4']
        prompts = [('prompt1', AgentType.USER, 'good', [], 'idx1'), ('prompt2', AgentType.USER, 'bad', ['O1'], 'idx2'), ('prompt3', AgentType.USER, 'bad', ['O2'], 'idx3'), ('prompt3', AgentType.USER, 'good', [], 'idx4')]
        expected_result = {
            "num_matches": 3,
            "num_mismatches": 1,
            "num_category_mismatches": 1,
            "mismatched_indices": ['idx4'],
            "mismatched_categories_indices": ['idx3']
        }
        result = parse_results(results, prompts)
        self.assertDictEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()