import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import numpy as np

load_dotenv()

st.title("ChatGPT-like clone")

# openai_api_key = "sk-proj-YSELmKxl31KEEydnjzE4T3BlbkFJWGWqCrqOv32OsMEyoa3T"

def get_response(user_query, chat_history):
    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

        Chat history: {chat_history}

        User question: {user_question}
        """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    chain = prompt | llm | StrOutputParser()
    return chain.stream({
            "chat_history": chat_history,
            "user_question": user_query,
        })

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt, "chart": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(get_response(prompt, st.session_state.messages))
        
    st.session_state.messages.append({"role": "assistant", 
                                    "content": response, 
                                    "chart": None})