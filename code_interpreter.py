from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent

load_dotenv()


def main():
    print("Start...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]


    # ========== Python agent executor =========
    
    python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
        tools=tools,
    )

    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    user_command_python = """generate and save in 'outputs' directory 2 qrcodes that point to www.google.com, 
    you have qrcode package already"""

    # python_agent_executor.invoke({"input": user_command_python})


     # ========== CSV agent executor =========

    csv_agent_executor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        path="docs/episode_info.csv",
        verbose=True,
        allow_dangerous_code=True
    )

    user_command_csv = """how many columns are in file episode_info.csv"""

    csv_agent_executor.invoke({"input": user_command_csv})


if __name__ == "__main__":
    main()
