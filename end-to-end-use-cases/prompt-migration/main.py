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
    
    target_lm = dspy.LM(
        model="together_ai/meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        api_key=os.getenv("TOGETHER_API_KEY")
    )
    # To run it with ollama
    # target_lm = dspy.LM('ollama_chat/llama3.2:3b-instruct-fp16', api_base='http://localhost:11434', api_key='')

    # To run it with huggingface
    # target_lm = dspy.HFModel(model="gpt2")
    
    engine = PromptMigrationEngine(openai_lm, target_lm)
    
    source_prompt = PromptTemplate(
        template="""You are an advanced Large Language Model tasked with generating Python code snippets in response to user prompts. Your primary objective is to provide accurate, concise, and well-structured Python functions. Follow these guidelines:

    Understand the Context: Analyze the input prompt and identify its category (e.g., API Usage, File Handling, Error Handling).

    Generate Code:
        Write Python code that directly addresses the user's request.
        Ensure the code is syntactically correct, functional, and adheres to Python best practices.
        Include necessary imports and handle potential edge cases.

    Error Handling:
        Include appropriate error handling where applicable (e.g., try-except blocks).
        If exceptions occur, provide meaningful error messages.

    Readability:
        Use clear variable names and include comments where necessary for clarity.
        Prioritize readability and maintainability in all generated code.

    Complexity Alignment:
        Tailor the code's complexity based on the indicated difficulty (e.g., simple, medium, complex).
        Ensure that the solution is neither overly simplistic nor unnecessarily complicated.

    Prompt Type:
        Focus on the code_generation type for creating Python functions.
        Avoid deviating from the task unless additional clarification is requested.

    Testing and Validity:
        Assume the function might be run immediately. Provide code that is ready for use or minimal adaptation.
        Highlight any dependencies or external libraries required.
        """,
        input_variables=["text"],
        model_type="openai"
    )
    
    eval_dataset = get_evaluation_dataset()


    # To evaluate on a specific subset, use the following:
    code_generation_dataset = get_eval_subset(prompt_type="code_generation")
    #simple_tasks = get_eval_subset(complexity="simple")
    evaluator = PromptEvaluator(openai_lm, target_lm)

    metrics = evaluator.evaluate(
        source_prompt.template,  # Same prompt for both
        source_prompt.template,  # Same prompt for both
        code_generation_dataset
    )
    
    print(f"Evaluation metrics:")
    print(f"  Accuracy: {metrics.accuracy:.2f}")
    print(f"  Similarity: {metrics.similarity:.2f}")
    print(f"  Consistency: {metrics.consistency:.2f}")
    
    # Migrate prompt
    print("Migrating prompt...")
    migrated_prompt = engine.migrate_prompt(source_prompt, code_generation_dataset)
    
    # Evaluate migration
    print("Evaluating migration...")
    metrics = evaluator.evaluate(
        source_prompt.template,
        migrated_prompt.template,
        code_generation_dataset
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