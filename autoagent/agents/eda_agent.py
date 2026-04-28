"""
EDA Agent
Profiles the dataset and generates Claude-narrated insights.
"""

import json
import os
import traceback

import anthropic
import numpy as np
import pandas as pd

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-20250514"


# ── Pandas helpers ──────────────────────────────────────────────────────────────

def _load_dataset(file_path: str) -> pd.DataFrame:
    ext = file_path.rsplit(".", 1)[-1].lower()
    loaders = {"csv": pd.read_csv, "xlsx": pd.read_excel, "json": pd.read_json}
    loader = loaders.get(ext, pd.read_csv)
    return loader(file_path)


def _basic_stats(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "numeric_stats": json.loads(numeric.describe().to_json()) if not numeric.empty else {},
        "categorical_stats": {
            c: {"unique": int(df[c].nunique()), "top": str(df[c].mode()[0]) if not df[c].empty else "N/A"}
            for c in categorical.columns
        },
    }


def _missing_analysis(df: pd.DataFrame) -> dict:
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    return {
        col: {"count": int(missing[col]), "pct": float(pct[col])}
        for col in df.columns
        if missing[col] > 0
    }


def _correlation_analysis(df: pd.DataFrame, target: str) -> dict:
    numeric = df.select_dtypes(include="number")
    if target in numeric.columns:
        corr = numeric.corr()[target].drop(target, errors="ignore")
        top = corr.abs().nlargest(10)
        return {col: round(float(corr[col]), 4) for col in top.index}
    return {}


def _outlier_analysis(df: pd.DataFrame) -> dict:
    result = {}
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        Q1 = numeric[col].quantile(0.25)
        Q3 = numeric[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((numeric[col] < Q1 - 1.5 * IQR) | (numeric[col] > Q3 + 1.5 * IQR)).sum()
        if outliers > 0:
            result[col] = {"outlier_count": int(outliers), "pct": round(float(outliers / len(df) * 100), 2)}
    return result


def _class_balance(df: pd.DataFrame, target: str, task_type: str) -> dict:
    if task_type == "classification" and target in df.columns:
        counts = df[target].value_counts()
        return {str(k): int(v) for k, v in counts.items()}
    return {}


# ── Tool definitions for Claude ─────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_eda_summary",
        "description": "Returns the full EDA summary including stats, missing values, correlations, outliers, and class balance.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
]


# ── Agent ───────────────────────────────────────────────────────────────────────

def run_eda_agent(state: dict) -> dict:
    messages = [{"role": "assistant", "content": "🔍 **EDA Agent** starting — loading your dataset..."}]

    try:
        df = _load_dataset(state["file_path"])
        target = state["target_column"]
        task_type = state["task_type"]

        eda_data = {
            "basic_stats": _basic_stats(df),
            "missing": _missing_analysis(df),
            "correlations": _correlation_analysis(df, target),
            "outliers": _outlier_analysis(df),
            "class_balance": _class_balance(df, target, task_type),
        }

        # Claude conversation with tool use
        system = (
            "You are an expert ML data scientist performing exploratory data analysis. "
            "When the tool is called, analyse the results thoroughly and produce a structured "
            "narrative covering: dataset overview, data quality issues, key patterns, "
            "correlations with the target, outliers, and concrete preprocessing recommendations. "
            "Be specific, actionable, and concise."
        )

        user_msg = (
            f"Problem: {state['problem_description']}\n"
            f"Target column: {target} | Task: {task_type}\n\n"
            "Please call get_eda_summary and then provide your full EDA analysis."
        )

        # Agentic loop
        convo = [{"role": "user", "content": user_msg}]
        claude_insight = ""

        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=system,
                tools=TOOLS,
                messages=convo,
            )

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use" and block.name == "get_eda_summary":
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(eda_data),
                        })

                convo.append({"role": "assistant", "content": response.content})
                convo.append({"role": "user", "content": tool_results})

            elif response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        claude_insight = block.text
                break
            else:
                break

        eda_report = {**eda_data, "claude_insight": claude_insight}

        messages.append({"role": "assistant", "content": f"✅ **EDA Complete**\n\n{claude_insight}"})

        return {
            "df_json": df.to_json(orient="split"),
            "eda_report": eda_report,
            "messages": messages,
            "current_agent": "feature_agent",
            "error": None,
        }

    except Exception as e:
        err = f"EDA Agent error: {traceback.format_exc()}"
        return {"messages": messages + [{"role": "assistant", "content": f"❌ {err}"}], "error": err}
