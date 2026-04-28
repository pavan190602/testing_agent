"""
Model Selection Agent
Trains multiple models, evaluates via cross-validation, explains with SHAP,
and lets Claude recommend the best approach.
"""

import json
import os
import traceback
import warnings

warnings.filterwarnings("ignore")

import anthropic
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import label_binarize

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-20250514"


# ── Model registry ──────────────────────────────────────────────────────────────

def _get_models(task_type: str) -> dict:
    if task_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42,
                                               eval_metric="logloss", verbosity=0)
        if HAS_LGB:
            models["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    else:
        models = {
            "Ridge Regression": Ridge(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        if HAS_LGB:
            models["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    return models


# ── Evaluation ──────────────────────────────────────────────────────────────────

def _evaluate_model(model, X_train, X_test, y_train, y_test, task_type: str, name: str) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task_type == "classification":
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
        result = {
            "name": name,
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
            "cv_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_std": round(float(cv_scores.std()), 4),
        }
        try:
            if len(np.unique(y_test)) == 2:
                result["roc_auc"] = round(float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])), 4)
        except Exception:
            pass
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        result = {
            "name": name,
            "r2": round(float(r2_score(y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "cv_r2_mean": round(float(cv_scores.mean()), 4),
            "cv_r2_std": round(float(cv_scores.std()), 4),
        }

    return result


# ── SHAP explainability ─────────────────────────────────────────────────────────

def _get_shap_importance(model, X_test, feature_names: list, task_type: str) -> dict:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:100])  # sample for speed

        if isinstance(shap_values, list):
            sv = np.abs(shap_values[1]).mean(axis=0)
        else:
            sv = np.abs(shap_values).mean(axis=0)

        importance = {feature_names[i]: round(float(sv[i]), 5) for i in range(len(feature_names))}
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15])
        return sorted_importance
    except Exception:
        # Fallback: feature importances from tree models
        try:
            fi = model.feature_importances_
            importance = {feature_names[i]: round(float(fi[i]), 5) for i in range(len(feature_names))}
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15])
        except Exception:
            return {}


# ── Tool definitions ────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_model_results",
        "description": "Returns comparison metrics for all trained models, SHAP feature importance, and training details.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
]


# ── Agent ───────────────────────────────────────────────────────────────────────

def run_model_agent(state: dict) -> dict:
    messages = [{"role": "assistant", "content": "🤖 **Model Selection Agent** starting — training models..."}]

    try:
        df = pd.read_json(state["df_engineered_json"], orient="split")
        target = state["target_column"]
        task_type = state["task_type"]
        feature_names = state.get("feature_names") or [c for c in df.columns if c != target]

        X = df[feature_names].values
        y = df[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = _get_models(task_type)
        results = []
        trained_models = {}

        for name, model in models.items():
            try:
                result = _evaluate_model(model, X_train, X_test, y_train, y_test, task_type, name)
                results.append(result)
                trained_models[name] = model
                messages.append({"role": "assistant", "content": f"  ✓ {name} trained"})
            except Exception as e:
                messages.append({"role": "assistant", "content": f"  ⚠️ {name} failed: {str(e)}"})

        # Pick best model
        if task_type == "classification":
            best = max(results, key=lambda r: r.get("cv_f1_mean", 0))
        else:
            best = max(results, key=lambda r: r.get("cv_r2_mean", -999))

        best_model = trained_models[best["name"]]

        # SHAP
        shap_importance = _get_shap_importance(best_model, X_test, feature_names, task_type)

        model_data = {
            "all_results": results,
            "best_model": best,
            "shap_importance": shap_importance,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_count": len(feature_names),
            "task_type": task_type,
        }

        # Claude recommendation
        system = (
            "You are a senior ML engineer presenting model evaluation results to a stakeholder. "
            "Call get_model_results, then write a clear recommendation covering: "
            "which model to deploy and why, key metrics explained in plain language, "
            "SHAP feature importance insights (what drives predictions), "
            "potential risks or limitations, and next steps for improvement. "
            "Be specific, authoritative, and actionable."
        )

        user_msg = (
            f"Problem: {state['problem_description']}\n"
            f"Target: {target} | Task: {task_type}\n\n"
            "Call get_model_results and provide your full model recommendation."
        )

        convo = [{"role": "user", "content": user_msg}]
        claude_insight = ""

        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2500,
                system=system,
                tools=TOOLS,
                messages=convo,
            )

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(model_data),
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

        model_report = {**model_data, "claude_insight": claude_insight}
        messages.append({"role": "assistant", "content": f"✅ **Model Selection Complete**\n\n{claude_insight}"})

        return {
            "model_report": model_report,
            "messages": messages,
            "current_agent": "done",
            "error": None,
        }

    except Exception as e:
        err = f"Model Agent error: {traceback.format_exc()}"
        return {"messages": messages + [{"role": "assistant", "content": f"❌ {err}"}], "error": err}
