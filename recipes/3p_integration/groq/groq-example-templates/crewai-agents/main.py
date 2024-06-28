import pandas as pd
import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq


def main():
    """
    Main function to initialize and run the CrewAI Machine Learning Assistant.

    This function sets up a machine learning assistant using the Llama 3 model with the ChatGroq API.
    It provides a text-based interface for users to define, assess, and solve machine learning problems
    by interacting with multiple specialized AI agents. The function outputs the results to the console 
    and writes them to a markdown file.

    Steps:
    1. Initialize the ChatGroq API with the specified model and API key.
    2. Display introductory text about the CrewAI Machine Learning Assistant.
    3. Create and configure four AI agents:
        - Problem_Definition_Agent: Clarifies the machine learning problem.
        - Data_Assessment_Agent: Evaluates the quality and suitability of the provided data.
        - Model_Recommendation_Agent: Suggests suitable machine learning models.
        - Starter_Code_Generator_Agent: Generates starter Python code for the project.
    4. Prompt the user to describe their machine learning problem.
    5. Check if a .csv file is available in the current directory and try to read it as a DataFrame.
    6. Define tasks for the agents based on user input and data availability.
    7. Create a Crew instance with the agents and tasks, and run the tasks.
    8. Print the results and write them to an output markdown file.
    """

    model = 'llama3-8b-8192'

    llm = ChatGroq(
            temperature=0, 
            groq_api_key = os.getenv('GROQ_API_KEY'), 
            model_name=model
        )

    print('CrewAI Machine Learning Assistant')
    multiline_text = """
    The CrewAI Machine Learning Assistant is designed to guide users through the process of defining, assessing, and solving machine learning problems. It leverages a team of AI agents, each with a specific role, to clarify the problem, evaluate the data, recommend suitable models, and generate starter Python code. Whether you're a seasoned data scientist or a beginner, this application provides valuable insights and a head start in your machine learning projects.
    """

    print(multiline_text)


    Problem_Definition_Agent = Agent(
        role='Problem_Definition_Agent',
        goal="""clarify the machine learning problem the user wants to solve, 
            identifying the type of problem (e.g., classification, regression) and any specific requirements.""",
        backstory="""You are an expert in understanding and defining machine learning problems. 
            Your goal is to extract a clear, concise problem statement from the user's input, 
            ensuring the project starts with a solid foundation.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Data_Assessment_Agent = Agent(
        role='Data_Assessment_Agent',
        goal="""evaluate the data provided by the user, assessing its quality, 
            suitability for the problem, and suggesting preprocessing steps if necessary.""",
        backstory="""You specialize in data evaluation and preprocessing. 
            Your task is to guide the user in preparing their dataset for the machine learning model, 
            including suggestions for data cleaning and augmentation.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Model_Recommendation_Agent = Agent(
        role='Model_Recommendation_Agent',
        goal="""suggest the most suitable machine learning models based on the problem definition 
            and data assessment, providing reasons for each recommendation.""",
        backstory="""As an expert in machine learning algorithms, you recommend models that best fit 
            the user's problem and data. You provide insights into why certain models may be more effective than others,
            considering classification vs regression and supervised vs unsupervised frameworks.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


    Starter_Code_Generator_Agent = Agent(
        role='Starter_Code_Generator_Agent',
        goal="""generate starter Python code for the project, including data loading, 
            model definition, and a basic training loop, based on findings from the problem definitions,
            data assessment and model recommendation""",
        backstory="""You are a code wizard, able to generate starter code templates that users 
            can customize for their projects. Your goal is to give users a head start in their coding efforts.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


    user_question = input("Describe your ML problem: ")
    data_upload = False
    # Check if there is a .csv file in the current directory
    if any(file.endswith(".csv") for file in os.listdir()):
        sample_fp = [file for file in os.listdir() if file.endswith(".csv")][0]
        try:
            # Attempt to read the uploaded file as a DataFrame
            df = pd.read_csv(sample_fp).head(5)

            # If successful, set 'data_upload' to True
            data_upload = True

            # Display the DataFrame in the app
            print("Data successfully uploaded and read as DataFrame:")
            print(df)
        except Exception as e:
            print(f"Error reading the file: {e}")

    if user_question:

        task_define_problem = Task(
        description="""Clarify and define the machine learning problem, 
            including identifying the problem type and specific requirements.

            Here is the user's problem:
            {ml_problem}
            """.format(ml_problem=user_question),
        agent=Problem_Definition_Agent,
        expected_output="A clear and concise definition of the machine learning problem."
        )

        if data_upload:
            task_assess_data = Task(
                description="""Evaluate the user's data for quality and suitability, 
                suggesting preprocessing or augmentation steps if needed.

                Here is a sample of the user's data:
                {df}
                The file name is called {uploaded_file}

                """.format(df=df.head(),uploaded_file=sample_fp),
                agent=Data_Assessment_Agent,
                expected_output="An assessment of the data's quality and suitability, with suggestions for preprocessing or augmentation if necessary."
            )
        else:
            task_assess_data = Task(
                description="""The user has not uploaded any specific data for this problem,
                but please go ahead and consider a hypothetical dataset that might be useful
                for their machine learning problem. 
                """,
                agent=Data_Assessment_Agent,
                expected_output="A hypothetical dataset that might be useful for the user's machine learning problem, along with any necessary preprocessing steps."
            )

        task_recommend_model = Task(
        description="""Suggest suitable machine learning models for the defined problem 
            and assessed data, providing rationale for each suggestion.""",
        agent=Model_Recommendation_Agent,
        expected_output="A list of suitable machine learning models for the defined problem and assessed data, along with the rationale for each suggestion."
        )


        task_generate_code = Task(
        description="""Generate starter Python code tailored to the user's project using the model recommendation agent's recommendation(s), 
            including snippets for package import, data handling, model definition, and training
            """,
        agent=Starter_Code_Generator_Agent,
        expected_output="Python code snippets for package import, data handling, model definition, and training, tailored to the user's project, plus a brief summary of the problem and model recommendations."
        )


        crew = Crew(
            agents=[Problem_Definition_Agent, Data_Assessment_Agent, Model_Recommendation_Agent,  Starter_Code_Generator_Agent], 
            tasks=[task_define_problem, task_assess_data, task_recommend_model,  task_generate_code], 
            verbose=False
        )

        result = crew.kickoff()

        print(result)

        with open('output.md', "w") as file:
            print('\n\nThese results have been exported to output.md')
            file.write(result)


if __name__ == "__main__":
    main()