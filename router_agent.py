from typing import Any

from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent


load_dotenv()


def main():
    print("Start...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    You have qrcode package installed
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
        tools=tools
    )

    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    csv_agent_executor = create_csv_agent( 
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        path="docs/episode_info.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    ################################ Router Agent ########################################################

    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        # print(original_prompt)
        return python_agent_executor.invoke({"input": original_prompt})

    def csv_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        # print(original_prompt)
        return csv_agent_executor.invoke({"input": original_prompt})

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""useful when you need to transform natural language to python and execute the python code,
                          returning the results of the code execution
                          DOES NOT ACCEPT CODE AS INPUT""",
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor_wrapper,
            description="""Use this tool to query and process data from the csv file using Python REPL.
            Takes natural language queries related to the CSV file and executes Python code to process the data""",
        ),
    ]

    router_instruction = """Based on the given input, delegate the task to the appropriate sub-agent based on the content of the input.
        If you are unsure which agent to use, respond with "I don't know"."""

    router_prompt = base_prompt.partial(instructions=router_instruction)
    router_agent = create_react_agent(
        prompt=router_prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
        tools=tools,
    )
    router_agent_executor = AgentExecutor(agent=router_agent, tools=tools, verbose=True)

    # print(router_agent_executor.invoke({"input": "which season has the most episodes from file?"}))

    print(router_agent_executor.invoke({
        "input": """generate and save in 'outputs' directory 2 qrcodes that point to `www.google.com`"""}))


if __name__ == "__main__":
    main()
