from typing import List

class PromptFormatter:
    @staticmethod
    def openai_to_llama(prompt: str) -> str:
        """Convert OpenAI-style prompts to Llama format."""
        # Basic conversion logic
        converted = prompt.replace("{{", "{").replace("}}", "}")
        return converted
    
    @staticmethod
    def extract_variables(prompt: str) -> List[str]:
        """Extract variable names from a prompt template."""
        import re
        pattern = r"\{([^}]+)\}"
        matches = re.findall(pattern, prompt)
        return list(set(matches)) 