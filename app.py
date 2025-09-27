'''
Multi Model Chatbot using Ollama and Langchain
also have LangSmith tracking
'''


import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="simple_chatbot"

st.title("MultiModal Chatbot using Ollama")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,llm):
     llm = ChatOllama(model=llm)
     output_parser = StrOutputParser()
     chain = prompt|llm|output_parser
     answer = chain.invoke({'question':question})
     return answer

st.sidebar.title("Settings")
 
llm = st.sidebar.selectbox("Select the model: ",["gemma3:1b","qwen3:4b"])


st.write("Go ahead and ask any question")
user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input,llm)
    st.write(response)
else:
    st.write("Please provide the query")
