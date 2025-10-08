import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict, Literal
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLCheckerTool
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# Load env
load_dotenv()
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Database and LLM
encoded_password = quote_plus(DB_PASSWORD)
engine = create_engine(f'mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}/{DB_NAME}')
db = SQLDatabase(engine=engine)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Graph logic
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    correct_or_not: str

system_message = """
Given an input question, create a syntactically correct {dialect} query to run to help find the answer.
Unless specified, limit your query to at most {top_k} results.
Only use the following tables:
{table_info}
"""
user_prompt = "Question: {input}"
query_prompt_template = ChatPromptTemplate([("system", system_message), ("user", user_prompt)])

class QueryOutput(TypedDict):
    query: Annotated[str, ..., "SQL query."]

def Generate(state: State):
    prompt = query_prompt_template.invoke({"dialect": db.dialect, "top_k": 10, "table_info": db.get_table_info(), "input": state["question"]})
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def Execute(state: State):
    tool = QuerySQLDatabaseTool(db=db)
    result = tool.invoke(state["query"])
    return {"result": result}

class Routes(BaseModel):
    route: Literal["Generate", "Execute"] = Field(description="Decide")

router = llm.with_structured_output(Routes)

def Check(state: State):
    tool = QuerySQLCheckerTool(db=db, llm=llm)
    checked = tool.invoke({"query": state["query"]})
    routing = router.invoke([SystemMessage(content="Decide"), HumanMessage(content=checked)])
    return {"correct_or_not": routing.route}

def next_route(state: State):
    return "Generate" if state["correct_or_not"] == "Generate" else "Execute"

def Answer(state: State):
    prompt = f"Question: {state['question']}\nSQL Query: {state['query']}\nSQL Result: {state['result']}"
    response = llm.invoke(prompt)
    return {"answer": response.content}

graph = StateGraph(State)
graph.add_node("Generate", Generate)
graph.add_node("Check", Check)
graph.add_node("Execute", Execute)
graph.add_node("Answer", Answer)
graph.add_edge(START, "Generate")
graph.add_edge("Generate", "Check")
graph.add_conditional_edges("Check", next_route, {"Generate": "Generate", "Execute": "Execute"})
graph.add_edge("Execute", "Answer")
graph.add_edge("Answer", END)
graph_builder = graph.compile()

# Streamlit UI
st.set_page_config(page_title="SQL Assistant", page_icon="🧠", layout="wide")
st.title("🧠 LLM-Powered SQL Assistant")

question = st.text_input("Ask your question about the database:")

if st.button("Run Query"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            input_state = {"question": question}
            result = graph_builder.invoke(input_state)
            st.subheader("✅ Answer")
            st.write(result["answer"])
            st.divider()
            st.text_area("🔍 Generated SQL Query", value=result.get("query", ""), height=100)
            st.text_area("📊 SQL Result", value=str(result.get("result", "")), height=200)
