import dspy
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    template: str
    input_variables: List[str]
    model_type: str  # 'openai' or 'llama'

class PromptMigrationEngine:
    def __init__(self, source_lm: dspy.LM, target_lm: dspy.LM):
        self.source_lm = source_lm
        self.target_lm = target_lm
        dspy.configure(lm=source_lm)
    
    def _optimize_transformation(self, transformer, eval_dataset):
        """Optimize the transformation using the evaluation dataset."""
        class PromptQualityMetric:
            def __init__(self, source_lm, target_lm):
                self.source_lm = source_lm
                self.target_lm = target_lm
            
            def __call__(self, example, prediction, trace=None):
                if not hasattr(prediction, 'target'):
                    return 0.0
                
                try:
                    # Get outputs from both models using the prompts
                    source_output = self.source_lm(example.source)
                    target_output = self.target_lm(prediction.target)
                    
                    # Compare outputs (basic similarity)
                    from difflib import SequenceMatcher
                    similarity = SequenceMatcher(None, 
                                              str(source_output), 
                                              str(target_output)).ratio()
                    return similarity
                except Exception as e:
                    print(f"Error in metric: {e}")
                    return 0.0
        
        optimizer = dspy.BootstrapFewShotWithRandomSearch(
            metric=PromptQualityMetric(self.source_lm, self.target_lm),
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
            num_threads=1
        )
        
        # Prepare training data
        train_data = []
        for item in eval_dataset:
            # Create example with both prompt and expected output
            example = dspy.Example(
                source=item["text"],
                expected_output=item["expected_answer"]
            ).with_inputs("source")
            train_data.append(example)
        
        return optimizer.compile(transformer, trainset=train_data)
    
    def migrate_prompt(self, 
                      source_prompt: PromptTemplate,
                      eval_dataset: Optional[List[Dict]] = None) -> PromptTemplate:
        """Migrates a prompt from source LM to target LM format."""
        
        class PromptTransformation(dspy.Signature):
            """Convert a prompt from one format to another."""
            source = dspy.InputField(desc="Source prompt template")
            target = dspy.OutputField(desc="Transformed prompt template that maintains functionality while adapting to target model format")
        
        class Transformer(dspy.Module):
            def __init__(self):
                super().__init__()
                self.chain = dspy.ChainOfThought(PromptTransformation)
            
            def forward(self, source):
                # Add context about the transformation task
                prompt = f"""
                Transform this prompt while:
                1. Maintaining core functionality
                2. Adapting to target model format
                3. Preserving input variables
                4. Keeping essential instructions
                
                Source prompt:
                {source}
                """
                return self.chain(source=prompt)
        
        transformer = Transformer()
        
        if eval_dataset:
            transformer = self._optimize_transformation(transformer, eval_dataset)
            
        result = transformer(source=source_prompt.template)
        
        # Format for target model
        if source_prompt.model_type == "openai" and "llama" in str(self.target_lm):
            result.target = f"### Instruction:\n{result.target}\n\n### Response:"
        
        return PromptTemplate(
            template=result.target,
            input_variables=source_prompt.input_variables,
            model_type='llama'
        ) 