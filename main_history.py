from typing import TypedDict
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import os

from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Connect DB + LLM
encoded_password = quote_plus(DB_PASSWORD)
engine = create_engine(f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}/{DB_NAME}")
db = SQLDatabase(engine=engine)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Thread ID (per chat)
THREAD_ID = "001"

# ---- Define State ----
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    chat_history: list  # store last 2 user-bot turns

# ---- Prompts ----
system_msg = """You are an intelligent SQL assistant.
Use the chat history for context and create a syntactically correct {dialect} query using only tables from: {table_info}.
If user refers to something mentioned before, infer it from context."""
user_prompt = "Chat History:\n{history}\n\nQuestion: {input}"

query_prompt = ChatPromptTemplate([
    ("system", system_msg),
    ("user", user_prompt)
])

class QueryOutput(TypedDict):
    query: str

# ---- Node 1: Generate SQL ----
def Generate(state: State):
    # Build context from last 2 messages
    history_text = "\n".join(state.get("chat_history", [])[-4:])  # last 2 exchanges = 4 messages (2 user + 2 bot)
    
    prompt = query_prompt.invoke({
        "dialect": db.dialect,
        "table_info": db.get_table_info(),
        "history": history_text,
        "input": state["question"]
    })
    
    structured = llm.with_structured_output(QueryOutput)
    state["query"] = structured.invoke(prompt)["query"]
    return {"query": state["query"]}

# ---- Node 2: Execute SQL ----
def Execute(state: State):
    state["result"] = QuerySQLDatabaseTool(db=db).invoke(state["query"])
    return {"result": state["result"]}

# ---- Node 3: Generate Final Answer ----
def Answer(state: State):
    prompt = f"""Question: {state['question']}
SQL: {state['query']}
Result: {state['result']}
Only display the answer briefly."""
    answer = llm.invoke(prompt).content
    # update chat history
    history = state.get("chat_history", [])
    history += [f"User: {state['question']}", f"Bot: {answer}"]
    state["chat_history"] = history[-4:]  # keep last 2 turns only
    state["answer"] = answer
    return {"answer": state["answer"], "chat_history": state["chat_history"]}

# ---- Memory + Graph ----
memory = InMemorySaver()
graph = StateGraph(State)
graph.add_node("Generate", Generate)
graph.add_node("Execute", Execute)
graph.add_node("Answer", Answer)
graph.add_edge(START, "Generate")
graph.add_edge("Generate", "Execute")
graph.add_edge("Execute", "Answer")
graph.add_edge("Answer", END)

graph = graph.compile(checkpointer=memory)

# ---- Config ----
config = {"configurable": {"thread_id": THREAD_ID}}

print("💬 SQL Assistant started! Type 'exit' to quit.\n")

chat_history = []

while True:
    q = input("🧠 You: ")
    if q.lower() in ["exit", "quit"]:
        break

    state = {"question": q, "chat_history": chat_history}
    result = graph.invoke(state, config=config)
    chat_history = result["chat_history"]
    print("🤖 Bot:", result["answer"])

print("\nSession Ended ✅")
