from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the endpoint from the environment variable
ollama_endpoint = os.getenv("OLLAMA_HOST")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3", endpoint=ollama_endpoint)

chain = prompt | model

res = chain.invoke({"question": "What is LangChain?"})
print(res)