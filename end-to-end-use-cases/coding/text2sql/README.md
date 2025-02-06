## Text2SQL: Natural Language to SQL Interface

This project provides a set of scripts to convert natural language queries into SQL statements using Meta's Llama model. The goal is to enable users to interact with databases using natural language inputs, making it easier for non-technical users to access and analyze data. 

For detailed instructions on setting up the environment, creating a database, and executing natural language queries using the Text2SQL interface, please refer to the quickstart.ipynb notebook.

### Structure:

- quickstart.ipynb: A Quick Demo of Text2SQL Using Llama 3.3. This Jupyter Notebook includes examples of how to use the interface to execute natural language queries on the sample data. It uses Llama 3.3 to answer questions about a SQLite database using LangChain and the Llama cloud provider Together.ai.
- nba.txt: A text file containing NBA roster information, which is used as sample data for demonstration purposes.
- txt2csv.py: A script that converts text data into a CSV format. This script is used to preprocess the input data before it is fed into csv2db.py.
- csv2db.py: A script that imports data from a CSV file into a SQLite database. This script is used to populate the database with sample data.
- nba_roster.db: A SQLite database file created from the nba.txt data, used to test the Text2SQL interface.

### Detailed steps on running the notebook:

- Before getting started, please make sure to setup Together.ai and get an API key from [here](https://www.together.ai/). 

- First, please install the requirements from [here](https://github.com/meta-llama/llama-cookbook/blob/main/end-to-end-use-cases/coding/text2sql/requirements.txt) by running inside the folder:

```
git clone https://github.com/meta-llama/llama-cookbook.git
cd llama-cookbook/end-to-end-use-cases/coding/text2sql/
pip install -r requirements.txt
```

### Contributing
Contributions are welcome! If you'd like to add new features or improve existing ones, please submit a pull request. We encourage contributions in the following areas:
- Adding support for additional databases
- Developing new interfaces or applications that use the Text2SQL interface