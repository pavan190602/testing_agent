"""
Feature Engineering Agent
Applies smart preprocessing and lets Claude explain every decision.
"""

import json
import os
import traceback

import anthropic
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-20250514"


# ── Preprocessing helpers ───────────────────────────────────────────────────────

def _handle_missing(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list]:
    steps = []
    for col in df.columns:
        if col == target:
            df = df.dropna(subset=[col])
            continue
        missing_pct = df[col].isnull().mean()
        if missing_pct == 0:
            continue
        if missing_pct > 0.5:
            df = df.drop(columns=[col])
            steps.append(f"Dropped '{col}' — {missing_pct:.0%} missing")
        elif df[col].dtype in ["object", "category"]:
            df[col] = df[col].fillna(df[col].mode()[0])
            steps.append(f"Filled '{col}' (categorical) with mode")
        else:
            strategy = "median" if df[col].skew() > 1 else "mean"
            fill_val = df[col].median() if strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_val)
            steps.append(f"Filled '{col}' (numeric) with {strategy} ({fill_val:.3f})")
    return df, steps


def _encode_categoricals(df: pd.DataFrame, target: str, task_type: str) -> tuple[pd.DataFrame, list]:
    steps = []
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target in cat_cols:
        cat_cols.remove(target)

    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique == 2:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            steps.append(f"Binary-encoded '{col}' (2 unique values)")
        elif n_unique <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            steps.append(f"One-hot encoded '{col}' ({n_unique} categories → {len(dummies.columns)} cols)")
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            steps.append(f"Label-encoded '{col}' ({n_unique} unique values — high cardinality)")

    # Encode target for classification
    if task_type == "classification" and df[target].dtype == "object":
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target].astype(str))
        steps.append(f"Label-encoded target '{target}'")

    return df, steps


def _scale_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list]:
    steps = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    for col in numeric_cols:
        skew = df[col].skew()
        if abs(skew) > 2:
            scaler = RobustScaler()
            steps.append(f"RobustScaler on '{col}' (skew={skew:.2f})")
        else:
            scaler = StandardScaler()
            steps.append(f"StandardScaler on '{col}' (skew={skew:.2f})")
        df[col] = scaler.fit_transform(df[[col]])

    return df, steps


def _remove_low_variance(df: pd.DataFrame, target: str, threshold: float = 0.01) -> tuple[pd.DataFrame, list]:
    steps = []
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
    variances = df[numeric_cols].var()
    to_drop = variances[variances < threshold].index.tolist()
    if to_drop:
        df = df.drop(columns=to_drop)
        steps.append(f"Removed low-variance features: {to_drop}")
    return df, steps


# ── Tool definitions ────────────────────────────────────────────────────────────

def _build_tools(steps_summary: dict) -> list:
    return [
        {
            "name": "get_feature_engineering_results",
            "description": "Returns the complete feature engineering transformation log with all decisions made.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
    ]


# ── Agent ───────────────────────────────────────────────────────────────────────

def run_feature_agent(state: dict) -> dict:
    messages = [{"role": "assistant", "content": "⚙️ **Feature Engineering Agent** starting..."}]

    try:
        df = pd.read_json(state["df_json"], orient="split")
        target = state["target_column"]
        task_type = state["task_type"]

        # Run transformations
        df, missing_steps = _handle_missing(df, target)
        df, encode_steps = _encode_categoricals(df, target, task_type)
        df, scale_steps = _scale_features(df, target)
        df, variance_steps = _remove_low_variance(df, target)

        all_steps = missing_steps + encode_steps + scale_steps + variance_steps
        feature_names = [c for c in df.columns if c != target]

        steps_summary = {
            "missing_handling": missing_steps,
            "encoding": encode_steps,
            "scaling": scale_steps,
            "variance_removal": variance_steps,
            "final_features": feature_names,
            "final_shape": list(df.shape),
        }

        # Claude narrates decisions
        system = (
            "You are a senior ML engineer explaining feature engineering decisions to a data science team. "
            "When the tool is called, write a clear, structured explanation of every transformation applied, "
            "why each was chosen, and what impact it will have on model training. "
            "Also note any potential concerns or alternative approaches worth considering."
        )

        user_msg = (
            f"Problem: {state['problem_description']}\n"
            f"Target: {target} | Task: {task_type}\n\n"
            "Call get_feature_engineering_results and explain all the feature engineering decisions."
        )

        convo = [{"role": "user", "content": user_msg}]
        claude_insight = ""

        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=system,
                tools=_build_tools(steps_summary),
                messages=convo,
            )

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(steps_summary),
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

        feature_report = {**steps_summary, "claude_insight": claude_insight}
        messages.append({"role": "assistant", "content": f"✅ **Feature Engineering Complete**\n\n{claude_insight}"})

        return {
            "df_engineered_json": df.to_json(orient="split"),
            "feature_names": feature_names,
            "feature_report": feature_report,
            "messages": messages,
            "current_agent": "model_agent",
            "error": None,
        }

    except Exception as e:
        err = f"Feature Agent error: {traceback.format_exc()}"
        return {"messages": messages + [{"role": "assistant", "content": f"❌ {err}"}], "error": err}
