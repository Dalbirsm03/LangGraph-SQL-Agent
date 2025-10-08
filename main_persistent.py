import streamlit as st
from typing import TypedDict, Annotated
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import os
import json

from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# ----------------- PersistentDict -----------------
class PersistentDict(dict):
    def __init__(self, filepath, format="json"):
        super().__init__()
        self.filepath = filepath    
        self.format = format
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                if self.format == "json":
                    data = json.load(f)
                    self.update(data)

    def sync(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            if self.format == "json":
                json.dump(self, f, indent=2)

# ----------------- Load environment -----------------
load_dotenv()
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ----------------- SQL connection -----------------
encoded_password = quote_plus(DB_PASSWORD)
engine = create_engine(f'mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}/{DB_NAME}')
db = SQLDatabase(engine=engine)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# ----------------- Persistent memory -----------------
THREAD_ID = "001"
memory_file = f"memory_{THREAD_ID}.json"
memory = PersistentDict(memory_file)

# ----------------- State -----------------
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

# ----------------- Prompts -----------------
system_message = """
Given an input question, create a syntactically correct {dialect} query to help find the answer.
Only use the following tables:
{table_info}
"""
user_prompt = "Question: {input}"
query_prompt_template = ChatPromptTemplate([("system", system_message), ("user", user_prompt)])

class QueryOutput(TypedDict):
    query: Annotated[str, ..., "SQL query."]

# ----------------- Graph Nodes -----------------
def Generate(state: State):
    prompt = query_prompt_template.invoke({
        "dialect": db.dialect,
        "table_info": db.get_table_info(),
        "input": state["question"]
    })
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    state["query"] = result["query"]
    return {"query": result["query"]}

def Execute(state: State):
    tool = QuerySQLDatabaseTool(db=db)
    result = tool.invoke(state["query"])
    state["result"] = result
    return {"result": result}

def Answer(state: State):
    prompt = f"Question: {state['question']}\nSQL Query: {state['query']}\nSQL Result: {state['result']}"
    response = llm.invoke(prompt)
    state["answer"] = response.content
    # Save state to memory immediately
    memory[len(memory)] = state.copy()
    memory.sync()
    return {"answer": response.content}

# ----------------- Build Graph -----------------
graph = StateGraph(State)
graph.add_node("Generate", Generate)
graph.add_node("Execute", Execute)
graph.add_node("Answer", Answer)
graph.add_edge(START, "Generate")
graph.add_edge("Generate", "Execute")
graph.add_edge("Execute", "Answer")
graph.add_edge("Answer", END)

graph_builder = graph.compile()

# ----------------- Streamlit UI -----------------
st.title("SQL Assistant with Persistent Memory")

question = st.text_input("Enter your SQL question:")

if st.button("Ask") and question:
    input_state = {"question": question}
    result = graph_builder.invoke(input_state)
    st.write("Answer:", result["answer"])

# Show full conversation history
st.subheader("Conversation History")
for i, snapshot in memory.items():
    st.write(f"Step {i}:")
    st.json(snapshot)

# Reset memory
if st.button("Reset Thread"):
    memory.clear()
    memory.sync()
    st.success("Memory cleared for this thread!")
