from typing import List, Dict

def get_evaluation_dataset() -> List[Dict]:
    """
    Returns a comprehensive evaluation dataset for testing prompt migrations.
    Each test case includes:
    - text: Input text
    - expected_answer: Expected output
    - prompt_type: Type of prompt (summarization, classification, qa, etc.)
    - complexity: Difficulty level (simple, medium, complex)
    """
    return [
        # Summarization examples
        {
            "text": "The quick brown fox jumps over the lazy dog.",
            "expected_answer": "A fox jumps over a dog.",
            "prompt_type": "summarization",
            "complexity": "simple"
        },
        {
            "text": """Machine learning is a subset of artificial intelligence that focuses on developing 
                   systems that can learn from and make decisions based on data. It has numerous 
                   applications in various fields including healthcare, finance, and autonomous vehicles.""",
            "expected_answer": "Machine learning is an AI technology that enables systems to learn and make decisions from data, used in healthcare, finance, and autonomous vehicles.",
            "prompt_type": "summarization",
            "complexity": "medium"
        },

        # Classification examples
        {
            "text": "I absolutely loved this product! Best purchase ever!",
            "expected_answer": "Positive",
            "prompt_type": "sentiment_classification",
            "complexity": "simple"
        },
        {
            "text": "The product works fine but the customer service could be better.",
            "expected_answer": "Neutral",
            "prompt_type": "sentiment_classification",
            "complexity": "medium"
        },

        # Question-Answering examples
        {
            "text": "What is the capital of France? Context: Paris is the capital and largest city of France.",
            "expected_answer": "Paris",
            "prompt_type": "qa",
            "complexity": "simple"
        },
        {
            "text": """What causes rain? Context: Rain is precipitation of liquid water in the form of droplets. 
                   Water vapor in warm air rises and cools, forming clouds. When the droplets become too 
                   heavy, they fall as rain.""",
            "expected_answer": "Rain occurs when water vapor in warm air rises, cools to form clouds, and droplets become heavy enough to fall.",
            "prompt_type": "qa",
            "complexity": "medium"
        },

        # Code-related examples
        {
            "text": "Write a function to add two numbers in Python.",
            "expected_answer": "def add(a, b):\n    return a + b",
            "prompt_type": "code_generation",
            "complexity": "simple"
        },
        {
            "text": "Explain what this code does: for i in range(len(arr)): arr[i] *= 2",
            "expected_answer": "This code multiplies each element in the array 'arr' by 2.",
            "prompt_type": "code_explanation",
            "complexity": "simple"
        },

        # Text transformation examples
        {
            "text": "convert this to passive voice: The cat chased the mouse.",
            "expected_answer": "The mouse was chased by the cat.",
            "prompt_type": "text_transformation",
            "complexity": "simple"
        },
        {
            "text": "translate to French: Hello, how are you?",
            "expected_answer": "Bonjour, comment allez-vous?",
            "prompt_type": "translation",
            "complexity": "simple"
        },

        # Complex reasoning examples
        {
            "text": """A train leaves Station A at 2:00 PM traveling at 60 mph. Another train leaves 
                   Station B at 3:00 PM traveling at 75 mph in the opposite direction. If the stations 
                   are 375 miles apart, at what time will the trains meet?""",
            "expected_answer": "The trains will meet at 5:00 PM.",
            "prompt_type": "problem_solving",
            "complexity": "complex"
        },
        {
            "text": """Analyze the environmental impact of electric vehicles versus traditional 
                   gasoline vehicles, considering manufacturing, operation, and disposal.""",
            "expected_answer": """Electric vehicles typically have higher manufacturing emissions but lower 
                              operational emissions compared to gasoline vehicles. Overall lifecycle 
                              environmental impact depends on electricity source and battery recycling.""",
            "prompt_type": "analysis",
            "complexity": "complex"
        },

        # Simple Code Generation
        {
            "text": "Write a Python function to check if a number is prime.",
            "expected_answer": """def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True""",
            "prompt_type": "code_generation",
            "complexity": "medium"
        },
        {
            "text": "Create a Python function to reverse a string.",
            "expected_answer": """def reverse_string(s):
    return s[::-1]""",
            "prompt_type": "code_generation",
            "complexity": "simple"
        },
        
        # Code Explanation
        {
            "text": "Explain what this code does: [x*x for x in range(10) if x % 2 == 0]",
            "expected_answer": "This list comprehension creates a list of squares of even numbers from 0 to 9. It filters numbers where x modulo 2 equals 0 (even numbers) and squares them.",
            "prompt_type": "code_explanation",
            "complexity": "medium"
        },
        
        # Algorithm Implementation
        {
            "text": "Write a Python function to implement binary search.",
            "expected_answer": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1""",
            "prompt_type": "code_generation",
            "complexity": "medium"
        },
        
        # Data Structure Implementation
        {
            "text": "Implement a Stack class in Python using a list.",
            "expected_answer": """class Stack:
    def __init__(self):
        self.items = []
        
    def push(self, item):
        self.items.append(item)
        
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        
    def is_empty(self):
        return len(self.items) == 0
        
    def peek(self):
        if not self.is_empty():
            return self.items[-1]""",
            "prompt_type": "code_generation",
            "complexity": "medium"
        },
        
        # Code Debugging
        {
            "text": "Find and fix the bug in this code: def factorial(n): return n * factorial(n-1)",
            "expected_answer": """def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n-1)""",
            "prompt_type": "code_debugging",
            "complexity": "medium"
        },
        
        # Code Optimization
        {
            "text": "Optimize this code: def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2) if n > 1 else n",
            "expected_answer": """def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b""",
            "prompt_type": "code_optimization",
            "complexity": "medium"
        },
        
        # API Usage
        {
            "text": "Write a Python function using requests to fetch data from a REST API endpoint.",
            "expected_answer": """import requests

def fetch_data(url, params=None):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None""",
            "prompt_type": "code_generation",
            "complexity": "medium"
        },
        
        # File Handling
        {
            "text": "Write a Python function to read a CSV file and return it as a list of dictionaries.",
            "expected_answer": """import csv

def read_csv(file_path):
    data = []
    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None""",
            "prompt_type": "code_generation",
            "complexity": "medium"
        },
        
        # Error Handling
        {
            "text": "Write a Python function that safely converts a string to integer with error handling.",
            "expected_answer": """def safe_int_convert(s):
    try:
        return int(s), None
    except ValueError as e:
        return None, str(e)""",
            "prompt_type": "code_generation",
            "complexity": "simple"
        },
        
        # Complex Algorithm
        {
            "text": "Implement a Python function for Depth-First Search on a graph.",
            "expected_answer": """def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    
    for next_node in graph[start]:
        if next_node not in visited:
            dfs(graph, next_node, visited)
            
    return visited""",
            "prompt_type": "code_generation",
            "complexity": "complex"
        }
    ]

def get_eval_subset(prompt_type: str = None, complexity: str = None) -> List[Dict]:
    """
    Returns a filtered subset of the evaluation dataset based on prompt type and/or complexity.
    
    Args:
        prompt_type: Type of prompts to filter (e.g., 'summarization', 'qa', etc.)
        complexity: Complexity level to filter (e.g., 'simple', 'medium', 'complex')
    """
    dataset = get_evaluation_dataset()
    
    if prompt_type:
        dataset = [d for d in dataset if d["prompt_type"] == prompt_type]
    
    if complexity:
        dataset = [d for d in dataset if d["complexity"] == complexity]
    
    return dataset 