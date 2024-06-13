# Executing Verified Queries with Function Calling

A command line application that allows users to ask questions about their DuckDB data using the Groq API. The application uses function calling to find the most similar pre-verified query to the user's question, execute it against the data, and return the results.

## Features

- **Function Calling**: The application uses function calling to match the user's question to the most relevant pre-verified SQL query.

- **SQL Execution**: The application executes the selected SQL query on a DuckDB database and displays the result.

## Functions

- `get_verified_queries(directory_path)`: Reads YAML files from the specified directory and loads the verified SQL queries and their descriptions.

- `execute_duckdb_query_function_calling(query_name, verified_queries_dict)`: Executes the provided SQL query using DuckDB and returns the result as a DataFrame.

## Data

The application queries data from CSV files located in the data folder:

- `employees.csv`: Contains employee data including their ID, full name, and email address.

- `purchases.csv`: Records purchase details including purchase ID, date, associated employee ID, amount, and product name.

## Verified Queries

The verified SQL queries and their descriptions are stored in YAML files located in the `verified-queries` folder. Descriptions are used to semantically map prompts to queries:

- `most-recent-purchases.yaml`: Returns the 5 most recent purchases

- `most-expensive-purchase.yaml`: Finds the most expensive purchases

- `number-of-teslas.yaml`: Counts the number of Teslas purchased

- `employees-without-purchases.yaml`: Gets employees without any recent purchases

## Usage

<!-- markdown-link-check-disable -->

You will need to store a valid Groq API Key as a secret to proceed with this example. You can generate one for free [here](https://console.groq.com/keys).

<!-- markdown-link-check-enable -->

You can [fork and run this application on Replit](https://replit.com/@GroqCloud/Execute-Verified-SQL-Queries-with-Function-Calling) or run it on the command line with `python main.py`.

## Customizing with Your Own Data

This application is designed to be flexible and can be easily customized to work with your own data. If you want to use your own data, follow these steps:

1. **Replace the CSV files**: The application queries data from CSV files located in the `data` folder. Replace these files with your own CSV files.

2. **Modify the verified queries**: The verified SQL queries and their descriptions are stored in YAML files located in the `verified-queries` folder. Replace these files with your own verified SQL queries and descriptions.
