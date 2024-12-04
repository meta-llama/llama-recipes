import dspy
from prompt_migration.engine import PromptMigrationEngine, PromptTemplate
from prompt_migration.evaluator import PromptEvaluator
from prompt_migration.eval_dataset import get_evaluation_dataset, get_eval_subset

import os
import dotenv

dotenv.load_dotenv()

def main():
    openai_lm = dspy.LM(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # target_lm = dspy.LM(
    #     model="together_ai/togethercomputer/meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    #     api_key=os.getenv("TOGETHER_API_KEY")
    # )
    # target_lm = dspy.LM('ollama_chat/llama3.2:3b-instruct-fp16', api_base='http://localhost:11434', api_key='')
    target_lm = dspy.HFModel(model="gpt2")
    
    engine = PromptMigrationEngine(openai_lm, target_lm)
    
    source_prompt = PromptTemplate(
        template="Write a Python function that takes as input a file path to an image, loads the image into memory as a numpy array, then crops the rows and columns around the perimeter if they are darker than a threshold value. Use the mean value of rows and columns to decide if they should be marked for deletion.",
        input_variables=["text"],
        model_type="openai"
    )
    
    eval_dataset = get_evaluation_dataset()


    # To evaluate on a specific subset, use the following:
    #summarization_dataset = get_eval_subset(prompt_type="summarization")
    #simple_tasks = get_eval_subset(complexity="simple")
    
    # Migrate prompt
    print("Migrating prompt...")
    migrated_prompt = engine.migrate_prompt(source_prompt, eval_dataset)
    
    # Evaluate migration
    print("Evaluating migration...")
    evaluator = PromptEvaluator(openai_lm, target_lm)
    metrics = evaluator.evaluate(
        source_prompt.template,
        migrated_prompt.template,
        eval_dataset
    )
    
    print(f"\nResults:")
    print(f"Original prompt: {source_prompt.template}")
    print(f"Migrated prompt: {migrated_prompt.template}")
    print(f"Evaluation metrics:")
    print(f"  Accuracy: {metrics.accuracy:.2f}")
    print(f"  Similarity: {metrics.similarity:.2f}")
    print(f"  Consistency: {metrics.consistency:.2f}")

if __name__ == "__main__":
    main() 