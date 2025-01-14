# 'Groqing the Stock Market' with Llama 3 Function Calling

This is a simple application that leverages the yfinance API to provide insights into stocks and their prices. The application uses the Llama 3 model on Groq in conjunction with Langchain to call functions based on the user prompt.

## Key Functions

- **get_stock_info(symbol, key)**: This function fetches various information about a given stock symbol. The information can be anything from the company's address to its financial ratios. The 'key' parameter specifies the type of information to fetch.

- **get_historical_price(symbol, start_date, end_date)**: This function fetches the historical stock prices for a given symbol from a specified start date to an end date. The returned data is a DataFrame with the date and closing price of the stock.

- **plot_price_over_time(historical_price_dfs)**: This function takes a list of DataFrames (each containing historical price data for a stock) and plots the prices over time using Plotly. The plot is saved to the same directory as the app.

- **call_functions(llm_with_tools, user_prompt)**: This function takes the user's question, invokes the appropriate tool (either get_stock_info or get_historical_price), and generates a response. If the user asked for historical prices, it also calls plot_price_over_time to generate a plot.

## Function Calling

The function calling in this application is handled by the Groq API, abstracted with Langchain. When the user asks a question, the application invokes the appropriate tool with parameters based on the user's question. The tool's output is then used to generate a response.

## Usage

<!-- markdown-link-check-disable -->

You will need to store a valid Groq API Key as a secret to proceed with this example. You can generate one for free [here](https://console.groq.com/keys).

<!-- markdown-link-check-enable -->

You can [fork and run this application on Replit](https://replit.com/@GroqCloud/Groqing-the-Stock-Market-Function-Calling-with-Llama3) or run it on the command line with `python main.py`.
