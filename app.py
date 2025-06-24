import streamlit as st
import os
import json
from dotenv import load_dotenv
from typing import TypedDict, List
from serpapi import GoogleSearch
from langgraph.graph import StateGraph, END
from langchain_community.llms import HuggingFaceEndpoint

# Load environment variables
load_dotenv()

# LangGraph State Definition
class GraphState(TypedDict):
    input: str
    action: str
    output: str
    retry_count: int
    max_retries: int
    valid: bool

# Initialize Hugging Face LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # Change to your model if needed
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.7
)

# Node: Classify the intent
def classify_query(state: GraphState) -> GraphState:
    query = state["input"]
    return {
        "input": query,
        "action": "search" if "search" in query.lower() else "answer",
        "output": "",
        "retry_count": 0,
        "max_retries": 1,
        "valid": False
    }

# Node: Search tool using SerpAPI
def search_tool(state: GraphState) -> GraphState:
    search = GoogleSearch({
        "q": state["input"],
        "api_key": os.getenv("SERP_API_KEY"),
        "num": 3
    })
    results = search.get_dict()

    snippets = []
    if "organic_results" in results:
        for r in results["organic_results"][:3]:
            title = r.get("title", "No Title")
            link = r.get("link", "#")
            snippet = r.get("snippet", "")
            snippets.append(f"[{title}]({link})\n\n{snippet}")

    output = "\n\n---\n\n".join(snippets) if snippets else "No search results found."

    return {
        **state,
        "output": output,
        "valid": True
    }

# Node: Generate answer using HuggingFace LLM
def answer_tool(state: GraphState) -> GraphState:
    response = llm.invoke(state["input"])
    return {
        **state,
        "output": response
    }

# Node: Validate LLM output and retry if too short
def validate_output(state: GraphState) -> GraphState:
    if len(state["output"]) < 15 and state["retry_count"] < state["max_retries"]:
        return {
            **state,
            "retry_count": state["retry_count"] + 1,
            "valid": False
        }
    return {
        **state,
        "valid": True
    }

# Build LangGraph
graph = StateGraph(GraphState)
graph.add_node("classify", classify_query)
graph.add_node("search", search_tool)
graph.add_node("answer", answer_tool)
graph.add_node("validate", validate_output)

graph.set_entry_point("classify")
graph.add_conditional_edges("classify", lambda s: s["action"], {
    "search": "search",
    "answer": "answer"
})
graph.add_edge("search", END)
graph.add_edge("answer", "validate")
graph.add_conditional_edges("validate", lambda s: s["valid"], {
    True: END,
    False: "answer"
})

app_graph = graph.compile()

# Streamlit UI
st.set_page_config(page_title="Aarambh GPT", layout="centered")
st.title("Aarambh GPT")

if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[dict] = []

user_input = st.text_input("Bring it on")

if st.button("Send") and user_input.strip() != "":
    with st.spinner("Thinking..."):
        result = app_graph.invoke({
            "input": user_input,
            "action": "",
            "output": "",
            "retry_count": 0,
            "max_retries": 1,
            "valid": False
        })
        st.session_state.chat_history.append({
            "user": user_input,
            "bot": result["output"]
        })

# Chat utility buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
with col2:
    if st.session_state.chat_history:
        json_str = json.dumps(st.session_state.chat_history, indent=2)
        st.download_button("Download Chat", json_str, "chat_history.json", "application/json")

# Display chat history
st.subheader("Conversation History")
for turn in st.session_state.chat_history[::-1]:
    st.markdown(f"**You:** {turn['user']}")
    st.markdown(f"**GPT:** {turn['bot']}")
    st.markdown("---")
