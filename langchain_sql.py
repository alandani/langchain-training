from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

db = SQLDatabase.from_uri("sqlite:///data/chinook.db")

# check that the database has been instantiated correctly
db.get_usable_table_names()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# agent_executor.invoke("How many employees are there?")

agent_executor.invoke("List the total sales per country. Which country's customers spent the most?")
