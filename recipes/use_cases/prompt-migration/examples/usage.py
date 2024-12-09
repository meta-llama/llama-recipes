import dspy
from prompt_migration.engine import PromptMigrationEngine, PromptTemplate
from prompt_migration.evaluator import PromptEvaluator

# Initialize LMs
openai_lm = dspy.OpenAI(model="gpt-3.5-turbo")
target_lm = dspy.HFModel(model="gpt2")

# Create migration engine
engine = PromptMigrationEngine(openai_lm, target_lm)

# Define source prompt
source_prompt = PromptTemplate(
    template="Summarize the following text: {text}",
    input_variables=["text"],
    model_type="openai"
)

eval_dataset = [
    {"text": "Example text 1", "expected_answer": "Summary 1"},
    {"text": "Example text 2", "expected_answer": "Summary 2"},
]

# Migrate prompt
migrated_prompt = engine.migrate_prompt(source_prompt, eval_dataset)

# Evaluate migration
evaluator = PromptEvaluator(openai_lm, target_lm)
metrics = evaluator.evaluate(
    source_prompt.template,
    migrated_prompt.template,
    eval_dataset
)

print(f"Migrated prompt: {migrated_prompt.template}")
print(f"Evaluation metrics: {metrics}") 