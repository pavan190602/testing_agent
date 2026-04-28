"""
AutoAgent — Streamlit UI
Conversational ML engineering interface with real-time agent progress.
"""

import io
import json
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AutoAgent — Autonomous ML Engineering",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.agent-message {
    background: #f0f4ff;
    border-left: 4px solid #4f46e5;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-size: 14px;
}
.metric-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.best-badge {
    background: #dcfce7;
    color: #166534;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
}
.stProgress > div > div > div { background: #4f46e5; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🤖 AutoAgent")
    st.caption("Autonomous ML Engineering System")
    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Get yours at console.anthropic.com",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("1. 🔍 EDA Agent")
    st.markdown("2. ⚙️ Feature Engineer")
    st.markdown("3. 🤖 Model Selector")
    st.divider()
    st.markdown("**Stack**")
    st.caption("LangGraph · Claude API · scikit-learn · XGBoost · SHAP · Plotly")


# ── Header ────────────────────────────────────────────────────────────────────────

st.title("🤖 AutoAgent")
st.markdown("**Give it a dataset. It does the entire ML engineering job.**")
st.divider()


# ── Input Section ─────────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "json"],
        help="CSV, Excel, or JSON file",
    )

with col2:
    problem_description = st.text_area(
        "Problem Description",
        placeholder="e.g. Predict whether a customer will churn based on usage patterns and demographics.",
        height=100,
    )

col3, col4, col5 = st.columns(3)
with col3:
    target_column = st.text_input("Target Column", placeholder="e.g. churn")
with col4:
    task_type = st.selectbox("Task Type", ["classification", "regression"])
with col5:
    st.markdown("&nbsp;")
    run_button = st.button("🚀 Run AutoAgent", type="primary", use_container_width=True)


# ── Sample dataset loader ─────────────────────────────────────────────────────────

with st.expander("📂 No dataset? Use a sample"):
    sample_choice = st.selectbox(
        "Choose sample",
        ["Titanic (classification)", "Boston Housing (regression)", "Iris (classification)"],
    )
    load_sample = st.button("Load sample dataset")

    if load_sample:
        if "Titanic" in sample_choice:
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            try:
                df_sample = pd.read_csv(url)
                buf = io.BytesIO()
                df_sample.to_csv(buf, index=False)
                buf.seek(0)
                st.session_state["sample_bytes"] = buf.getvalue()
                st.session_state["sample_name"] = "titanic.csv"
                st.session_state["sample_target"] = "Survived"
                st.session_state["sample_task"] = "classification"
                st.success("Titanic dataset loaded! Set Target=Survived, Task=classification")
            except Exception:
                st.error("Could not fetch sample. Please upload your own CSV.")
        elif "Boston" in sample_choice:
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing(as_frame=True)
            df_s = data.frame
            buf = io.BytesIO()
            df_s.to_csv(buf, index=False)
            buf.seek(0)
            st.session_state["sample_bytes"] = buf.getvalue()
            st.session_state["sample_name"] = "housing.csv"
            st.session_state["sample_target"] = "MedHouseVal"
            st.session_state["sample_task"] = "regression"
            st.success("Housing dataset loaded! Set Target=MedHouseVal, Task=regression")
        else:
            from sklearn.datasets import load_iris
            data = load_iris(as_frame=True)
            df_s = data.frame
            df_s["target_name"] = data.target
            buf = io.BytesIO()
            df_s.to_csv(buf, index=False)
            buf.seek(0)
            st.session_state["sample_bytes"] = buf.getvalue()
            st.session_state["sample_name"] = "iris.csv"
            st.session_state["sample_target"] = "target_name"
            st.session_state["sample_task"] = "classification"
            st.success("Iris dataset loaded! Set Target=target_name, Task=classification")


# ── Run pipeline ──────────────────────────────────────────────────────────────────

if run_button:
    # Validate
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    file_bytes = None
    file_name = None

    if uploaded_file:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name
    elif "sample_bytes" in st.session_state:
        file_bytes = st.session_state["sample_bytes"]
        file_name = st.session_state["sample_name"]
        if not target_column:
            target_column = st.session_state.get("sample_target", "")

    if not file_bytes:
        st.error("Please upload a dataset or load a sample.")
        st.stop()
    if not target_column:
        st.error("Please specify the target column.")
        st.stop()
    if not problem_description:
        problem_description = f"Predict {target_column} from the provided features."

    # Save to temp file
    suffix = "." + file_name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Run pipeline
    st.divider()
    st.subheader("🔄 Pipeline Running")

    progress = st.progress(0, text="Starting pipeline...")
    chat_container = st.container()

    with chat_container:
        status_area = st.empty()

    all_messages = []

    def show_message(msg):
        all_messages.append(msg["content"])
        with chat_container:
            for m in all_messages:
                st.markdown(f'<div class="agent-message">{m}</div>', unsafe_allow_html=True)

    try:
        from orchestrator import run_pipeline

        # Stream progress
        progress.progress(10, text="🔍 EDA Agent running...")
        show_message({"content": "🔍 **EDA Agent** — Analysing your dataset..."})

        final_state = run_pipeline(
            file_path=tmp_path,
            problem_description=problem_description,
            target_column=target_column,
            task_type=task_type,
        )

        # Show all agent messages
        for msg in final_state.get("messages", []):
            show_message(msg)

        progress.progress(100, text="✅ Pipeline complete!")

        if final_state.get("error"):
            st.error(f"Pipeline error: {final_state['error']}")
        else:
            st.session_state["final_state"] = final_state
            st.session_state["df_json"] = final_state.get("df_json")
            st.session_state["target"] = target_column
            st.session_state["task_type"] = task_type
            st.success("✅ AutoAgent pipeline complete! See results below.")

    except Exception as e:
        st.error(f"Pipeline failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

    finally:
        os.unlink(tmp_path)


# ── Results Tabs ──────────────────────────────────────────────────────────────────

if "final_state" in st.session_state:
    st.divider()
    state = st.session_state["final_state"]

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 EDA Report", "⚙️ Features", "🤖 Models", "📊 SHAP Insights"])

    # ── TAB 1: EDA ──────────────────────────────────────────────────────────────

    with tab1:
        eda = state.get("eda_report", {})
        if eda:
            st.subheader("📊 Dataset Overview")
            basic = eda.get("basic_stats", {})
            shape = basic.get("shape", [0, 0])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{shape[0]:,}")
            c2.metric("Columns", shape[1])
            c3.metric("Missing Cols", len(eda.get("missing", {})))
            c4.metric("Outlier Cols", len(eda.get("outliers", {})))

            col_a, col_b = st.columns(2)

            with col_a:
                # Missing values chart
                missing = eda.get("missing", {})
                if missing:
                    st.subheader("Missing Values")
                    df_miss = pd.DataFrame(
                        [(k, v["pct"]) for k, v in missing.items()],
                        columns=["Column", "Missing %"]
                    )
                    fig = px.bar(df_miss, x="Column", y="Missing %",
                                 color="Missing %", color_continuous_scale="Reds")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("No missing values! 🎉")

            with col_b:
                # Correlations chart
                corr = eda.get("correlations", {})
                if corr:
                    st.subheader(f"Correlations with Target")
                    df_corr = pd.DataFrame(
                        list(corr.items()), columns=["Feature", "Correlation"]
                    ).sort_values("Correlation")
                    fig = px.bar(df_corr, x="Correlation", y="Feature", orientation="h",
                                 color="Correlation", color_continuous_scale="RdBu",
                                 color_continuous_midpoint=0)
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

            # Distribution of dataset - load raw df
            if st.session_state.get("df_json"):
                df_raw = pd.read_json(st.session_state["df_json"], orient="split")
                target = st.session_state.get("target")
                if target and target in df_raw.columns:
                    st.subheader("Target Distribution")
                    if st.session_state.get("task_type") == "classification":
                        fig = px.pie(df_raw, names=target, title=f"Distribution of {target}")
                    else:
                        fig = px.histogram(df_raw, x=target, nbins=40, title=f"Distribution of {target}")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

            # Claude insight
            st.subheader("🧠 Claude's Analysis")
            st.markdown(eda.get("claude_insight", "No insight available."))

    # ── TAB 2: Features ─────────────────────────────────────────────────────────

    with tab2:
        feat = state.get("feature_report", {})
        if feat:
            st.subheader("🔧 Transformations Applied")
            final_shape = feat.get("final_shape", [0, 0])
            c1, c2, c3 = st.columns(3)
            c1.metric("Final Features", feat.get("feature_count", len(feat.get("final_features", []))))
            c2.metric("Final Rows", final_shape[0])
            c3.metric("Final Cols", final_shape[1])

            col_a, col_b = st.columns(2)
            with col_a:
                if feat.get("missing_handling"):
                    st.markdown("**Missing Value Handling**")
                    for s in feat["missing_handling"]:
                        st.markdown(f"- {s}")
                if feat.get("variance_removal"):
                    st.markdown("**Low Variance Removal**")
                    for s in feat["variance_removal"]:
                        st.markdown(f"- {s}")

            with col_b:
                if feat.get("encoding"):
                    st.markdown("**Encoding**")
                    for s in feat["encoding"]:
                        st.markdown(f"- {s}")
                if feat.get("scaling"):
                    st.markdown("**Scaling**")
                    for s in feat["scaling"]:
                        st.markdown(f"- {s}")

            st.subheader("🧠 Claude's Explanation")
            st.markdown(feat.get("claude_insight", "No insight available."))

    # ── TAB 3: Models ───────────────────────────────────────────────────────────

    with tab3:
        model_r = state.get("model_report", {})
        if model_r:
            results = model_r.get("all_results", [])
            best_name = model_r.get("best_model", {}).get("name", "")

            st.subheader("📈 Model Comparison")

            if results:
                df_results = pd.DataFrame(results)
                task = st.session_state.get("task_type", "classification")

                # Highlight best model
                def highlight_best(row):
                    if row["name"] == best_name:
                        return ["background-color: #dcfce7"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    df_results.style.apply(highlight_best, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(f"🏆 Best model: **{best_name}** (highlighted in green)")

                # Chart
                if task == "classification":
                    metric = "cv_f1_mean"
                    label = "CV F1 Score (mean)"
                else:
                    metric = "cv_r2_mean"
                    label = "CV R² Score (mean)"

                if metric in df_results.columns:
                    colors = ["#4ade80" if n == best_name else "#6366f1" for n in df_results["name"]]
                    fig = go.Figure(go.Bar(
                        x=df_results["name"],
                        y=df_results[metric],
                        marker_color=colors,
                        text=df_results[metric].round(4),
                        textposition="outside",
                    ))
                    fig.update_layout(title=label, height=350, yaxis_title=label)
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("🧠 Claude's Recommendation")
            st.markdown(model_r.get("claude_insight", "No insight available."))

    # ── TAB 4: SHAP ─────────────────────────────────────────────────────────────

    with tab4:
        model_r = state.get("model_report", {})
        shap_imp = model_r.get("shap_importance", {})

        if shap_imp:
            st.subheader(f"🔍 SHAP Feature Importance — {model_r.get('best_model', {}).get('name', 'Best Model')}")
            st.caption("Mean absolute SHAP value — how much each feature drives predictions on average.")

            df_shap = pd.DataFrame(
                list(shap_imp.items()), columns=["Feature", "SHAP Value"]
            ).sort_values("SHAP Value")

            fig = px.bar(
                df_shap, x="SHAP Value", y="Feature", orientation="h",
                color="SHAP Value", color_continuous_scale="Viridis",
                title="Feature Importance (SHAP values)",
            )
            fig.update_layout(height=max(400, len(shap_imp) * 30))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Top 3 most influential features:**")
            top3 = list(shap_imp.items())[:3]
            for i, (feat, val) in enumerate(top3):
                st.markdown(f"{i+1}. **{feat}** — SHAP value: `{val}`")
        else:
            st.info("SHAP values not available for this model type.")


# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption("AutoAgent · Built with LangGraph + Claude API · github.com/pavan190602/autoagent")
