import dspy
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class EvaluationMetrics:
    accuracy: float
    similarity: float
    consistency: float

class PromptEvaluator:
    def __init__(self, source_lm: dspy.OpenAI, target_lm: dspy.LM):
        self.source_lm = source_lm
        self.target_lm = target_lm
        
    def _create_judge(self):
        """Create an LLM judge to evaluate prompt outputs."""
        class FactJudge(dspy.Signature):
            """Judge if the migrated prompt produces equivalent outputs."""
            source_output = dspy.InputField(desc="Output from source model")
            target_output = dspy.InputField(desc="Output from target model")
            factually_correct = dspy.OutputField(
                desc="Is the target output equivalent to the source output in terms of content and intent?",
                prefix="Factual[Yes/No]:"
            )
            reasoning = dspy.OutputField(desc="Explanation for the judgment")

        return dspy.ChainOfThought(FactJudge)

    def _get_model_output(self, model, text: str) -> str:
        """Helper function to get output from different model types."""
        try:
            # Try different methods since DSPy model interfaces can vary
            if hasattr(model, '__call__'):
                return model(text)
            elif hasattr(model, 'generate'):
                return model.generate(text)
            elif hasattr(model, 'complete'):
                return model.complete(text)
            else:
                raise AttributeError(f"Model {type(model)} has no supported generation method")
        except Exception as e:
            print(f"Error generating output with {type(model)}: {str(e)}")
            return ""

    def _calculate_metrics(self, evaluator, test_cases):
        """Calculate evaluation metrics using LLM as judge."""
        total_similarity = 0.0
        total_accuracy = 0.0
        total_consistency = 0.0
        
        judge = self._create_judge()
        
        for case in test_cases:
            source_output = self._get_model_output(self.source_lm, case["text"])
            target_output = self._get_model_output(self.target_lm, case["text"])
            
            judgment = judge(
                source_output=source_output,
                target_output=target_output
            )
            
            is_equivalent = judgment.factually_correct.lower() == "yes"
            
            similarity = float(is_equivalent)
            accuracy = float(target_output.lower() == case["expected_summary"].lower())
            consistency = float(is_equivalent)
            
            total_similarity += similarity
            total_accuracy += accuracy
            total_consistency += consistency
            
            print(f"\nJudge's reasoning: {judgment.reasoning}")
        
        n = len(test_cases)
        return EvaluationMetrics(
            accuracy=total_accuracy / n,
            similarity=total_similarity / n,
            consistency=total_consistency / n
        )
    
    def evaluate(self, 
                source_prompt: str, 
                target_prompt: str, 
                test_cases: List[Dict]) -> EvaluationMetrics:
        """Evaluates the quality of prompt migration using LLM as judge."""
        
        metrics = self._calculate_metrics(None, test_cases)  # evaluator param not needed anymore
        
        return metrics