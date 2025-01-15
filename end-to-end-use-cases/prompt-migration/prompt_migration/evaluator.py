import json
from typing import List, Dict
from dataclasses import dataclass
import dspy
import os
from datetime import datetime

@dataclass
class EvaluationMetrics:
    accuracy: float
    similarity: float
    consistency: float
    individual_scores: List[Dict]  # Store individual test case scores

class PromptEvaluator:
    def __init__(self, source_lm: dspy.LM, target_lm: dspy.LM):
        self.source_lm = source_lm
        self.target_lm = target_lm
        dspy.configure(lm=source_lm)  # Configure DSPy to use source_lm for judge
        
    def _create_judge(self):
        """Create an LLM judge to evaluate outputs."""
        class OutputJudge(dspy.Signature):
            """Judge the quality and equivalence of outputs."""
            input_text = dspy.InputField(desc="The coding task")
            source_output = dspy.InputField(desc="Output from source prompt")
            target_output = dspy.InputField(desc="Output from target prompt")
            expected_output = dspy.InputField(desc="Expected output from dataset")
            
            equivalence = dspy.OutputField(
                desc="Are the outputs functionally equivalent to the expected output? Answer ONLY with 'yes' or 'no'."
            )
            accuracy = dspy.OutputField(
                desc="Rate how well the outputs match the expected output. Provide ONLY a number between 0 and 100, no text."
            )
            consistency = dspy.OutputField(
                desc="Rate how consistent the outputs are with each other. Provide ONLY a number between 0 and 100, no text."
            )
            reasoning = dspy.OutputField(
                desc="Explain your evaluation, focusing on functionality and correctness."
            )

        class Judge(dspy.Module):
            def __init__(self):
                super().__init__()
                self.judge = dspy.ChainOfThought(OutputJudge)
            
            def forward(self, input_text, source_output, target_output, expected_output):
                try:
                    result = self.judge(
                        input_text=input_text,
                        source_output=source_output,
                        target_output=target_output,
                        expected_output=expected_output
                    )
                    
                    # Ensure numeric scores
                    def clean_score(score):
                        try:
                            # Extract just numbers
                            import re
                            numbers = re.findall(r'\d+', str(score))
                            return float(numbers[0]) if numbers else 0.0
                        except:
                            return 0.0
                    
                    result.accuracy = clean_score(result.accuracy)
                    result.consistency = clean_score(result.consistency)
                    result.equivalence = str(result.equivalence).lower().strip()
                    
                    return result
                except Exception as e:
                    print(f"Error in judge: {str(e)}")
                    return type('Result', (), {
                        'accuracy': '0',
                        'consistency': '0',
                        'equivalence': 'no',
                        'reasoning': f'Error in evaluation: {str(e)}'
                    })()

        return Judge()

    def _get_model_output(self, prompt: str, input_text: str) -> str:
        """Get output from target model using the provided prompt."""
        try:
            formatted_prompt = prompt.format(text=input_text)
            response = self.target_lm(formatted_prompt)
            
            if isinstance(response, list):
                return response[0] if response else ""
            return str(response)
        except Exception as e:
            print(f"Error generating output: {str(e)}")
            return ""

    def _calculate_metrics(self, source_prompt: str, target_prompt: str, test_cases: List[Dict]) -> EvaluationMetrics:
        """Calculate evaluation metrics using target model for both prompts."""
        total_similarity = 0.0
        total_accuracy = 0.0
        total_consistency = 0.0
        individual_scores = []
        
        judge = self._create_judge()
        num_cases = len(test_cases)
        
        for case in test_cases:
            input_text = case["text"]
            expected = case["expected_answer"]
            
            # Get outputs from target model using both prompts
            source_output = self._get_model_output(source_prompt, input_text)
            target_output = self._get_model_output(target_prompt, input_text)
            
            judgment = judge(
                input_text=input_text,
                source_output=source_output,
                target_output=target_output,
                expected_output=expected
            )
            
            accuracy_score = float(judgment.accuracy) / 100
            consistency_score = float(judgment.consistency) / 100
            is_equivalent = judgment.equivalence.lower() == "yes"
            
            case_scores = {
                "input": input_text,
                "expected": expected,
                "source_output": source_output,
                "target_output": target_output,
                "accuracy": accuracy_score,
                "consistency": consistency_score,
                "equivalent": is_equivalent,
                "reasoning": judgment.reasoning
            }
            individual_scores.append(case_scores)
            
            total_accuracy += accuracy_score
            total_consistency += consistency_score
            total_similarity += float(is_equivalent)
            
            print(f"\nEvaluation for case: {input_text[:50]}...")
            print(f"Source output: {source_output[:100]}...")
            print(f"Target output: {target_output[:100]}...")
            print(f"Expected: {expected[:100]}...")
            print(f"Judge's reasoning: {judgment.reasoning}")
            print(f"Scores - Accuracy: {accuracy_score:.2f}, Consistency: {consistency_score:.2f}, Equivalent: {is_equivalent}")
        
        metrics = EvaluationMetrics(
            accuracy=total_accuracy / num_cases,
            similarity=total_similarity / num_cases,
            consistency=total_consistency / num_cases,
            individual_scores=individual_scores
        )
        
        results = {
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "aggregate_metrics": {
                "accuracy": metrics.accuracy,
                "similarity": metrics.similarity,
                "consistency": metrics.consistency
            },
            "individual_scores": individual_scores
        }
        
        self._save_results(results)

        
        return metrics
    
    def evaluate(self, 
                source_prompt: str, 
                target_prompt: str, 
                test_cases: List[Dict]) -> EvaluationMetrics:
        """Evaluates both prompts using the target model."""
        return self._calculate_metrics(source_prompt, target_prompt, test_cases)
    
    def _save_results(self, results: dict, filename: str = 'results.json') -> None:
        """Save results to a JSON file with a new name if the file already exists."""

        if os.path.exists(filename):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{timestamp}{ext}"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")