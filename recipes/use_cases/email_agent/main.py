import argparse
import email_agent
from email_agent import *
from functions_prompt import system_prompt


def main():
    parser = argparse.ArgumentParser(description="Set email address")
    parser.add_argument("--email", type=str, required=True, help="Your Gmail address")
    args = parser.parse_args()

    email_agent.set_email_service(args.email)

    greeting = llama31("hello", "Your name is Email Agent, an assistant that can perform all email related tasks for your user.")
    agent_response = f"{greeting}\n\nYour ask: "
    agent = Agent(system_prompt)

    while True:
        ask = input(agent_response)
        if ask == "bye":
            print(llama31("bye"))
            break
        print("\n-------------------------\nCalling Llama...")
        agent(ask)
        agent_response = "Your ask: "


if __name__ == "__main__":
    main()



