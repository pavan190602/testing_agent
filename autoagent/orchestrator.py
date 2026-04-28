"""
AutoAgent — LangGraph Orchestrator
Coordinates EDA → Feature Engineering → Model Selection pipeline
"""

import operator
from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END

from agents.eda_agent import run_eda_agent
from agents.feature_agent import run_feature_agent
from agents.model_agent import run_model_agent


# ── State ──────────────────────────────────────────────────────────────────────

class AutoAgentState(TypedDict):
    # User inputs
    file_path: str
    problem_description: str
    target_column: str
    task_type: str                    # 'classification' | 'regression'

    # Serialised data (DataFrames stored as JSON strings)
    df_json: Optional[str]
    df_engineered_json: Optional[str]
    feature_names: Optional[list]

    # Agent reports
    eda_report: Optional[dict]
    feature_report: Optional[dict]
    model_report: Optional[dict]

    # UI streaming  — list is append-only across nodes
    messages: Annotated[list, operator.add]
    current_agent: str
    error: Optional[str]


# ── Node wrappers ───────────────────────────────────────────────────────────────

def eda_node(state: AutoAgentState) -> dict:
    return run_eda_agent(state)


def feature_node(state: AutoAgentState) -> dict:
    return run_feature_agent(state)


def model_node(state: AutoAgentState) -> dict:
    return run_model_agent(state)


def should_continue(state: AutoAgentState) -> str:
    """Route to END if any agent raised an error."""
    return END if state.get("error") else "continue"


# ── Graph ───────────────────────────────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(AutoAgentState)

    workflow.add_node("eda_agent", eda_node)
    workflow.add_node("feature_agent", feature_node)
    workflow.add_node("model_agent", model_node)

    workflow.set_entry_point("eda_agent")

    workflow.add_edge("eda_agent", "feature_agent")
    workflow.add_edge("feature_agent", "model_agent")
    workflow.add_edge("model_agent", END)

    return workflow.compile()


# Singleton graph
graph = build_graph()


def run_pipeline(
    file_path: str,
    problem_description: str,
    target_column: str,
    task_type: str,
) -> AutoAgentState:
    """
    Run the full AutoAgent pipeline and return the final state.
    """
    initial_state: AutoAgentState = {
        "file_path": file_path,
        "problem_description": problem_description,
        "target_column": target_column,
        "task_type": task_type,
        "df_json": None,
        "df_engineered_json": None,
        "feature_names": None,
        "eda_report": None,
        "feature_report": None,
        "model_report": None,
        "messages": [],
        "current_agent": "eda_agent",
        "error": None,
    }

    final_state = graph.invoke(initial_state)
    return final_state
