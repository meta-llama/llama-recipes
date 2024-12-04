import dspy
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    template: str
    input_variables: List[str]
    model_type: str  # 'openai' or 'llama'

class PromptMigrationEngine:
    def __init__(self, source_lm: dspy.OpenAI, target_lm: dspy.LM):
        self.source_lm = source_lm
        self.target_lm = target_lm
        dspy.configure(lm=source_lm)
    
    def _optimize_transformation(self, transformer, eval_dataset):
        """Optimize the transformation using the evaluation dataset."""
        class AccuracyMetric:
            def __call__(self, example, prediction, trace=None):
                return float(prediction.target == example.expected_output)
        
        optimizer = dspy.BootstrapFewShotWithRandomSearch(
            metric=AccuracyMetric(),
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            num_threads=4
        )
        
        train_data = [
            dspy.Example(
                source=item["text"],
                expected_output=item["expected_summary"]
            ).with_inputs("source") for item in eval_dataset
        ]
        
        return optimizer.compile(transformer, trainset=train_data)
    
    def migrate_prompt(self, 
                      source_prompt: PromptTemplate,
                      eval_dataset: Optional[List[Dict]] = None) -> PromptTemplate:
        """Migrates a prompt from source LM to target LM format."""
        
        class PromptTransformation(dspy.Signature):
            """Convert a prompt from one format to another."""
            source = dspy.InputField(desc="Source prompt template")
            target = dspy.OutputField(desc="Transformed prompt template")
        
        class Transformer(dspy.Module):
            def __init__(self):
                super().__init__()
                self.chain = dspy.ChainOfThought(PromptTransformation)
            
            def forward(self, source):
                return self.chain(source=source)
        
        transformer = Transformer()
        
        if eval_dataset:
            transformer = self._optimize_transformation(transformer, eval_dataset)
            
        result = transformer(source=source_prompt.template)
        
        return PromptTemplate(
            template=result.target,
            input_variables=source_prompt.input_variables,
            model_type='llama'
        ) 