import torch
from torch.nn.functional import softmax

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

"""
Utilities for loading the PromptGuard model and evaluating text for jailbreaks and indirect injections.
"""


def load_model_and_tokenizer(model_name='meta-llama/PromptGuard'):
    """
    Load the PromptGuard model from Hugging Face or a local model.
    
    Args:
        model_name (str): The name of the model to load. Default is 'meta-llama/PromptGuard'.
        
    Returns:
        transformers.PreTrainedModel: The loaded model.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_class_probabilities(model, tokenizer, text, temperature=1.0, device='cpu'):
    """
    Evaluate the model on the given text with temperature-adjusted softmax.
    
    Args:
        text (str): The input text to classify.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.
        
    Returns:
        torch.Tensor: The probability of each class adjusted by the temperature.
    """
    # Encode the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = inputs.to(device)
    # Get logits from the model
    with torch.no_grad():
        logits = model(**inputs).logits
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Apply softmax to get probabilities
    probabilities = softmax(scaled_logits, dim=-1)
    return probabilities


def get_jailbreak_score(model, tokenizer, text, temperature=1.0, device='cpu'):
    """
    Evaluate the probability that a given string contains malicious jailbreak or prompt injection.
    Appropriate for filtering dialogue between a user and an LLM.
    
    Args:
        text (str): The input text to evaluate.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.
        
    Returns:
        float: The probability of the text containing malicious content.
    """
    probabilities = get_class_probabilities(model, tokenizer, text, temperature, device)
    return probabilities[0, 2].item()


def get_indirect_injection_score(model, tokenizer, text, temperature=1.0, device='cpu'):
    """
    Evaluate the probability that a given string contains any embedded instructions (malicious or benign).
    Appropriate for filtering third party inputs (e.g. web searches, tool outputs) into an LLM.
    
    Args:
        text (str): The input text to evaluate.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.
        
    Returns:
        float: The combined probability of the text containing malicious or embedded instructions.
    """
    probabilities = get_class_probabilities(model, tokenizer, text, temperature, device)
    return (probabilities[0, 1] + probabilities[0, 2]).item()
