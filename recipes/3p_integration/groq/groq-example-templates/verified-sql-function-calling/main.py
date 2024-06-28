import os
from groq import Groq
import duckdb
import yaml
import glob
import json

def get_verified_queries(directory_path):
    """
    Reads YAML files from the specified directory, loads the verified SQL queries and their descriptions,
    and stores them in a dictionary.

    Parameters:
        directory_path (str): The path to the directory containing the YAML files with verified queries.

    Returns:
        dict: A dictionary where the keys are the names of the YAML files (without the directory path and file extension)
              and the values are the parsed content of the YAML files.
    """
    verified_queries_yaml_files = glob.glob(os.path.join(directory_path, '*.yaml'))
    verified_queries_dict = {}
    for file in verified_queries_yaml_files:
        with open(file, 'r') as stream:
            try:
                file_name = file[len(directory_path):-5]
                verified_queries_dict[file_name] = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                continue
        
    return verified_queries_dict


def execute_duckdb_query_function_calling(query_name,verified_queries_dict):
    """
    Executes a SQL query from the verified queries dictionary using DuckDB and returns the result as a DataFrame.

    Parameters:
        query_name (str): The name of the query to be executed, corresponding to a key in the verified queries dictionary.
        verified_queries_dict (dict): A dictionary containing verified queries, where the keys are query names and the values
                                      are dictionaries with query details including the SQL statement.

    Returns:
        pandas.DataFrame: The result of the executed query as a DataFrame.
    """
    
    original_cwd = os.getcwd()
    os.chdir('data')

    query = verified_queries_dict[query_name]['sql']
    
    try:
        conn = duckdb.connect(database=':memory:', read_only=False)
        query_result = conn.execute(query).fetchdf().reset_index(drop=True)
    finally:
        os.chdir(original_cwd)

    return query_result


model = "llama3-8b-8192"

# Initialize the Groq client
groq_api_key = os.getenv('GROQ_API_KEY')
client = Groq(
    api_key=groq_api_key
)

directory_path = 'verified-queries/'
verified_queries_dict = get_verified_queries(directory_path)

# Display the title and introduction of the application
multiline_text = """
Welcome! Ask questions about employee data or purchase details, like "Show the 5 most recent purchases" or "What was the most expensive purchase?". The app matches your question to pre-verified SQL queries for accurate results.
"""

print(multiline_text)

    
while True:
    # Get user input from the console
    user_input = input("You: ")

    
    #Simplify verified_queries_dict to just show query name and description
    query_description_mapping = {key: subdict['description'] for key, subdict in verified_queries_dict.items()}
    
    # Step 1: send the conversation and available functions to the model
    # Define the messages to be sent to the Groq API
    messages = [
        {
            "role": "system",
            "content": '''You are a function calling LLM that uses the data extracted from the execute_duckdb_query_function_calling function to answer questions around a DuckDB dataset.

            Extract the query_name parameter from this mapping by finding the one whose description best matches the user's question: 
            {query_description_mapping}
            '''.format(query_description_mapping=query_description_mapping)
        },
        {
            "role": "user",
            "content": user_input,
        }
    ]

    # Define the tool (function) to be used by the Groq API
    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_duckdb_query_function_calling",
                "description": "Executes a verified DuckDB SQL Query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_name": {
                            "type": "string",
                            "description": "The name of the verified query (i.e. 'most-recent-purchases')",
                        }
                    },
                    "required": ["query_name"],
                },
            },
        }
    ]

    # Send the conversation and available functions to the Groq API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",  
        max_tokens=4096
    )

    # Extract the response message and any tool calls from the response
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Define a dictionary of available functions
    available_functions = {
        "execute_duckdb_query_function_calling": execute_duckdb_query_function_calling,
    }

    # Iterate over the tool calls in the response
    for tool_call in tool_calls:
        function_name = tool_call.function.name  # Get the function name
        function_to_call = available_functions[function_name]  # Get the function to call
        function_args = json.loads(tool_call.function.arguments)  # Parse the function arguments
        print('Query found: ', function_args.get("query_name"))
        
        # Call the function with the provided arguments
        function_response = function_to_call(
            query_name=function_args.get("query_name"),
            verified_queries_dict=verified_queries_dict
        )

    # Print the function response (query result)
    print(function_response)

