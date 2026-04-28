"""
AutoAgent — Beginner-Friendly UI
Everything explained in plain English. No ML knowledge required.
"""

import io, os, json, tempfile, traceback
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoAgent — ML Made Easy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; }

/* Step indicator */
.step-pill {
    display: inline-block;
    background: #f1f5f9;
    border: 1.5px solid #e2e8f0;
    border-radius: 24px;
    padding: 6px 16px;
    font-size: 13px;
    font-weight: 600;
    color: #64748b;
    margin: 4px 4px 4px 0;
}
.step-pill.active {
    background: #eff6ff;
    border-color: #3b82f6;
    color: #1d4ed8;
}
.step-pill.done {
    background: #f0fdf4;
    border-color: #22c55e;
    color: #15803d;
}

/* Cards */
.explain-card {
    background: #fffbeb;
    border: 1.5px solid #fde68a;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 14px;
    line-height: 1.6;
}
.tip-card {
    background: #f0f9ff;
    border: 1.5px solid #bae6fd;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 14px;
}
.agent-card {
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px;
    margin: 10px 0;
    background: white;
}
.agent-running {
    border-color: #3b82f6;
    background: #eff6ff;
}
.agent-done {
    border-color: #22c55e;
    background: #f0fdf4;
}
.big-result {
    font-size: 28px;
    font-weight: 700;
    color: #1e293b;
    text-align: center;
    padding: 10px;
}
.result-label {
    font-size: 13px;
    color: #64748b;
    text-align: center;
    font-weight: 500;
}
.insight-box {
    background: #fafafa;
    border-left: 4px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 14px;
    line-height: 1.7;
    color: #1e293b;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🤖 AutoAgent")
st.markdown("**The AI that does your entire machine learning project for you.** No coding. No statistics. No experience needed.")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# WHAT IS THIS? — Always visible explainer
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🙋 What is AutoAgent? What does it do? (Read this first if you're new)", expanded=False):
    st.markdown("""
    ### In plain English — what is AutoAgent?

    Imagine you have a spreadsheet of data — maybe customer records, sales data, patient information, anything — and you want to **predict something** from that data.

    For example:
    - *"Will this customer cancel their subscription?"*
    - *"How much will this house sell for?"*
    - *"Will this patient be readmitted to hospital?"*

    Normally, answering these questions requires a **data scientist** who spends days or weeks:
    1. Exploring the data (called EDA — Exploratory Data Analysis)
    2. Cleaning and transforming it (Feature Engineering)
    3. Testing many machine learning models to find the best one
    4. Explaining what the model found

    **AutoAgent does all of that in minutes.** You upload your spreadsheet, tell it what you want to predict, and it runs three AI agents that handle each step — then explains everything in plain English using Claude (Anthropic's AI).

    ---

    ### The 3 agents, explained simply:

    | Agent | What it does | Like a person who... |
    |-------|-------------|---------------------|
    | 🔍 **EDA Agent** | Reads your data and finds patterns, problems, and interesting facts | ...looks through your spreadsheet and highlights everything important |
    | ⚙️ **Feature Agent** | Cleans the data and prepares it for AI | ...organises your messy files before handing them to an expert |
    | 🤖 **Model Agent** | Tests 5 different AI models and picks the best one | ...runs experiments and tells you which approach worked best |

    ---

    ### What do you need to use it?
    - A spreadsheet file (CSV or Excel)
    - An Anthropic API key (free to get, explained in Step 1 below)
    - That's it. No coding. No ML knowledge.
    """)

# ══════════════════════════════════════════════════════════════════════════════
# STEP TRACKER
# ══════════════════════════════════════════════════════════════════════════════
steps_done = st.session_state.get("steps_done", set())

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    cls = "done" if "api" in steps_done else ("active" if True else "")
    st.markdown(f'<span class="step-pill {cls}">{"✅" if "api" in steps_done else "1️⃣"} API Key</span>', unsafe_allow_html=True)
with col_s2:
    cls = "done" if "data" in steps_done else ("active" if "api" in steps_done else "")
    st.markdown(f'<span class="step-pill {cls}">{"✅" if "data" in steps_done else "2️⃣"} Upload Data</span>', unsafe_allow_html=True)
with col_s3:
    cls = "done" if "config" in steps_done else ("active" if "data" in steps_done else "")
    st.markdown(f'<span class="step-pill {cls}">{"✅" if "config" in steps_done else "3️⃣"} Configure</span>', unsafe_allow_html=True)
with col_s4:
    cls = "done" if "results" in steps_done else ("active" if "config" in steps_done else "")
    st.markdown(f'<span class="step-pill {cls}">{"✅" if "results" in steps_done else "4️⃣"} Results</span>', unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — API KEY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### Step 1 — Enter your Anthropic API Key")

with st.expander("❓ What is an API key and how do I get one?"):
    st.markdown("""
    An **API key** is like a password that lets AutoAgent use Claude (Anthropic's AI) to analyse your data and explain the results.

    **How to get one (free, takes 2 minutes):**
    1. Go to **console.anthropic.com**
    2. Click "Sign Up" and create a free account
    3. Once logged in, click **"API Keys"** in the left menu
    4. Click **"Create Key"** and give it any name (e.g. "autoagent")
    5. Copy the key — it starts with `sk-ant-...`
    6. Paste it in the box below

    > ⚠️ **Your key is private.** Never share it. It is used only on your computer and never stored anywhere.
    """)

saved_key = os.getenv("ANTHROPIC_API_KEY", "")
api_key = st.text_input(
    "Anthropic API Key",
    type="password",
    value=saved_key,
    placeholder="sk-ant-api03-...",
    help="Starts with sk-ant-. Get yours free at console.anthropic.com",
)

if api_key:
    os.environ["ANTHROPIC_API_KEY"] = api_key
    steps_done.add("api")
    st.session_state["steps_done"] = steps_done
    st.success("✅ API key saved!")
else:
    st.markdown('<div class="tip-card">💡 <strong>Tip:</strong> You can also create a <code>.env</code> file in the project folder with <code>ANTHROPIC_API_KEY=your-key-here</code> so you never have to type it again.</div>', unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — UPLOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### Step 2 — Upload your dataset")

with st.expander("❓ What kind of file do I need? What should be in it?"):
    st.markdown("""
    **What file format?**
    - CSV file (`.csv`) — the most common type. Can be exported from Excel, Google Sheets, etc.
    - Excel file (`.xlsx`)
    - JSON file (`.json`)

    **What should be in it?**
    Your file should look like a table — rows and columns. Each row is one record (one customer, one house, one patient). Each column is one piece of information (age, price, date, etc.).

    **Example — a customer churn dataset:**
    | customer_id | age | monthly_spend | contract_type | churned |
    |------------|-----|--------------|---------------|---------|
    | 001 | 34 | 45.50 | Monthly | Yes |
    | 002 | 56 | 89.00 | Annual | No |

    Here we would want to predict the `churned` column.

    **How many rows do I need?**
    At least 100 rows. The more the better — 500+ is ideal.

    **Don't have data?** Use our sample dataset below!
    """)

tab_upload, tab_sample = st.tabs(["📁 Upload my own file", "🎓 Use a sample dataset (no file needed)"])

file_bytes = None
file_name = None

with tab_upload:
    uploaded = st.file_uploader(
        "Drag and drop your file here, or click to browse",
        type=["csv", "xlsx", "json"],
        label_visibility="collapsed",
    )
    if uploaded:
        file_bytes = uploaded.read()
        file_name = uploaded.name
        steps_done.add("data")
        st.session_state["steps_done"] = steps_done
        df_preview = pd.read_csv(io.BytesIO(file_bytes)) if file_name.endswith(".csv") else pd.read_excel(io.BytesIO(file_bytes))
        st.success(f"✅ File loaded: **{file_name}** — {df_preview.shape[0]:,} rows, {df_preview.shape[1]} columns")
        st.markdown("**Preview (first 5 rows):**")
        st.dataframe(df_preview.head(), use_container_width=True)
        st.session_state["df_preview"] = df_preview
        st.session_state["file_bytes"] = file_bytes
        st.session_state["file_name"] = file_name

with tab_sample:
    st.markdown('<div class="explain-card">🎓 These are real public datasets used to teach machine learning. Perfect for trying AutoAgent for the first time.</div>', unsafe_allow_html=True)

    sample = st.radio(
        "Pick a sample dataset:",
        [
            "🚢 Titanic — Predict who survived the Titanic disaster (classification)",
            "🏠 California Housing — Predict house prices (regression)",
            "🌸 Iris Flowers — Classify flower species (classification)",
        ],
        label_visibility="collapsed",
    )

    if st.button("📥 Load this sample dataset", use_container_width=True):
        with st.spinner("Loading sample data..."):
            try:
                if "Titanic" in sample:
                    df_s = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
                    tgt, task = "Survived", "classification"
                    desc = "Predict whether a passenger survived the Titanic disaster based on their age, sex, ticket class, and other information."
                elif "Housing" in sample:
                    from sklearn.datasets import fetch_california_housing
                    raw = fetch_california_housing(as_frame=True)
                    df_s = raw.frame
                    tgt, task = "MedHouseVal", "regression"
                    desc = "Predict the median house value for California districts based on features like income, house age, and location."
                else:
                    from sklearn.datasets import load_iris
                    raw = load_iris(as_frame=True)
                    df_s = raw.frame
                    df_s["species"] = raw.target
                    tgt, task = "species", "classification"
                    desc = "Classify iris flowers into 3 species based on measurements of their petals and sepals."

                buf = io.BytesIO()
                df_s.to_csv(buf, index=False)
                buf.seek(0)
                file_bytes = buf.getvalue()
                file_name = f"{tgt}_dataset.csv"

                st.session_state["file_bytes"] = file_bytes
                st.session_state["file_name"] = file_name
                st.session_state["df_preview"] = df_s
                st.session_state["preset_target"] = tgt
                st.session_state["preset_task"] = task
                st.session_state["preset_desc"] = desc
                steps_done.add("data")
                st.session_state["steps_done"] = steps_done

                st.success(f"✅ Sample loaded: **{file_name}** — {df_s.shape[0]:,} rows, {df_s.shape[1]} columns")
                st.markdown("**Preview:**")
                st.dataframe(df_s.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Could not load sample: {e}")

# Restore from session if already loaded
if not file_bytes and "file_bytes" in st.session_state:
    file_bytes = st.session_state["file_bytes"]
    file_name = st.session_state["file_name"]

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CONFIGURE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### Step 3 — Tell AutoAgent what to predict")

with st.expander("❓ What do I fill in here?"):
    st.markdown("""
    **Target column** — This is the column you want to predict. It's the "answer" column in your data.
    - Example: if you want to predict whether a customer churns, your target column is probably called `churn` or `Churned`
    - Look at your data preview above and find the column name

    **Task type:**
    - Choose **Classification** if the answer is a category — Yes/No, A/B/C, 0/1, "spam"/"not spam"
    - Choose **Regression** if the answer is a number — house price, temperature, sales amount

    **Problem description** — Just describe in plain English what you're trying to do. AutoAgent uses this to give you better explanations.
    """)

df_preview = st.session_state.get("df_preview")
col_options = list(df_preview.columns) if df_preview is not None else []

col_a, col_b = st.columns(2)

with col_a:
    if col_options:
        preset_target = st.session_state.get("preset_target", col_options[-1])
        preset_idx = col_options.index(preset_target) if preset_target in col_options else len(col_options) - 1
        target_column = st.selectbox(
            "🎯 Target column — what do you want to predict?",
            col_options,
            index=preset_idx,
            help="Pick the column that contains the answer you want to predict",
        )
    else:
        target_column = st.text_input(
            "🎯 Target column — what do you want to predict?",
            placeholder="e.g. churn, price, survived",
        )

with col_b:
    preset_task = st.session_state.get("preset_task", "classification")
    task_type = st.selectbox(
        "📋 Task type",
        ["classification", "regression"],
        index=0 if preset_task == "classification" else 1,
        help="Classification = categories (Yes/No). Regression = numbers (prices, amounts).",
        format_func=lambda x: "📊 Classification — predict a category (Yes/No, A/B/C)" if x == "classification" else "📈 Regression — predict a number (price, amount, score)",
    )

problem_desc = st.text_area(
    "📝 Describe your problem in plain English (optional but recommended)",
    value=st.session_state.get("preset_desc", ""),
    placeholder="e.g. I want to predict whether a customer will cancel their subscription based on their usage history and demographics.",
    height=80,
)

if target_column and file_bytes:
    steps_done.add("config")
    st.session_state["steps_done"] = steps_done

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — RUN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### Step 4 — Run AutoAgent")

ready = api_key and file_bytes and target_column
if not ready:
    missing = []
    if not api_key: missing.append("API key")
    if not file_bytes: missing.append("dataset file")
    if not target_column: missing.append("target column")
    st.markdown(f'<div class="tip-card">⏳ Still needed before you can run: <strong>{", ".join(missing)}</strong></div>', unsafe_allow_html=True)

run_btn = st.button(
    "🚀 Run AutoAgent — Analyse My Data",
    type="primary",
    use_container_width=True,
    disabled=not ready,
)

if run_btn and ready:
    if not problem_desc:
        problem_desc = f"Predict the {target_column} column from the available features."

    suffix = "." + file_name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    st.divider()
    st.markdown("### ⏳ AutoAgent is working...")
    st.markdown('<div class="explain-card">The three AI agents are running in sequence. Each one will explain what it found in plain English. This usually takes 1–3 minutes depending on your dataset size.</div>', unsafe_allow_html=True)

    # Agent status display
    status_cols = st.columns(3)
    with status_cols[0]:
        eda_box = st.empty()
        eda_box.markdown('<div class="agent-card agent-running">🔍 <strong>EDA Agent</strong><br><small>Reading your data...</small></div>', unsafe_allow_html=True)
    with status_cols[1]:
        feat_box = st.empty()
        feat_box.markdown('<div class="agent-card">⚙️ <strong>Feature Agent</strong><br><small>Waiting...</small></div>', unsafe_allow_html=True)
    with status_cols[2]:
        model_box = st.empty()
        model_box.markdown('<div class="agent-card">🤖 <strong>Model Agent</strong><br><small>Waiting...</small></div>', unsafe_allow_html=True)

    progress_bar = st.progress(0, text="Starting pipeline...")
    log_area = st.empty()

    log_msgs = []
    def log(msg):
        log_msgs.append(msg)
        log_area.markdown("\n\n".join(log_msgs[-5:]))

    try:
        from orchestrator import run_pipeline

        log("🔍 EDA Agent is exploring your dataset...")
        progress_bar.progress(15, text="EDA Agent — reading and profiling your data...")

        final_state = run_pipeline(
            file_path=tmp_path,
            problem_description=problem_desc,
            target_column=target_column,
            task_type=task_type,
        )

        eda_box.markdown('<div class="agent-card agent-done">✅ <strong>EDA Agent</strong><br><small>Data analysis complete</small></div>', unsafe_allow_html=True)
        progress_bar.progress(50, text="Feature Agent — cleaning and transforming data...")
        log("⚙️ Feature Agent cleaned and prepared your data...")

        feat_box.markdown('<div class="agent-card agent-done">✅ <strong>Feature Agent</strong><br><small>Feature engineering complete</small></div>', unsafe_allow_html=True)
        progress_bar.progress(80, text="Model Agent — training and comparing models...")
        log("🤖 Model Agent trained 5 models and picked the best...")

        model_box.markdown('<div class="agent-card agent-done">✅ <strong>Model Agent</strong><br><small>Model selection complete</small></div>', unsafe_allow_html=True)
        progress_bar.progress(100, text="✅ Done!")

        if final_state.get("error"):
            st.error(f"Something went wrong: {final_state['error']}")
        else:
            st.session_state["final_state"] = final_state
            st.session_state["run_target"] = target_column
            st.session_state["run_task"] = task_type
            steps_done.add("results")
            st.session_state["steps_done"] = steps_done
            log_area.empty()
            st.success("✅ Done! Scroll down to see your results.")

    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.code(traceback.format_exc())
    finally:
        os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if "final_state" in st.session_state:
    state = st.session_state["final_state"]
    task = st.session_state.get("run_task", "classification")
    target = st.session_state.get("run_target", "target")

    st.divider()
    st.markdown("## 📊 Your Results")
    st.markdown('<div class="explain-card">Everything below was discovered and explained by AutoAgent\'s three AI agents. Each section includes a <strong>plain English explanation</strong> so you don\'t need any technical background to understand it.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔍 What's in my data?", "⚙️ How was it prepared?", "🤖 Which model won?"])


    # ── TAB 1: EDA ─────────────────────────────────────────────────────────────
    with tab1:
        eda = state.get("eda_report", {})
        basic = eda.get("basic_stats", {})
        shape = basic.get("shape", [0, 0])

        st.markdown("### At a glance")
        with st.expander("❓ What does this mean?"):
            st.markdown("These are the basic facts about your dataset — how big it is, how many pieces of information (columns) it has, and whether any data is missing.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total rows", f"{shape[0]:,}", help="Number of records in your dataset")
        c2.metric("Total columns", shape[1], help="Number of pieces of information per record")
        c3.metric("Columns with missing data", len(eda.get("missing", {})), help="Columns where some values are blank")
        c4.metric("Columns with outliers", len(eda.get("outliers", {})), help="Columns with extreme values that might affect the model")

        col_a, col_b = st.columns(2)

        with col_a:
            missing = eda.get("missing", {})
            if missing:
                st.markdown("#### ❗ Missing data")
                st.markdown("These columns have blank/missing values. The Feature Agent will fill these in automatically.")
                df_miss = pd.DataFrame([(k, v["pct"]) for k, v in missing.items()], columns=["Column", "% Missing"])
                fig = px.bar(df_miss, x="Column", y="% Missing", color="% Missing",
                             color_continuous_scale="Reds", title="How much data is missing per column")
                fig.update_layout(height=280, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("#### ✅ No missing data")
                st.success("Your dataset has no missing values — great quality data!")

        with col_b:
            corr = eda.get("correlations", {})
            if corr:
                st.markdown(f"#### 📈 What predicts `{target}`?")
                st.markdown("These columns are most correlated with what you're trying to predict. Positive = higher value means more likely. Negative = higher value means less likely.")
                df_corr = pd.DataFrame(list(corr.items()), columns=["Feature", "Correlation"]).sort_values("Correlation")
                fig = px.bar(df_corr, x="Correlation", y="Feature", orientation="h",
                             color="Correlation", color_continuous_scale="RdBu", color_continuous_midpoint=0,
                             title="Correlation with target")
                fig.update_layout(height=280)
                st.plotly_chart(fig, use_container_width=True)

        # Class balance
        balance = eda.get("class_balance", {})
        if balance and task == "classification":
            st.markdown("#### ⚖️ Target balance")
            st.markdown(f"How many rows belong to each category in `{target}`:")
            df_bal = pd.DataFrame(list(balance.items()), columns=["Category", "Count"])
            fig = px.pie(df_bal, names="Category", values="Count",
                         title=f"Distribution of '{target}'",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🧠 What the AI found")
        with st.expander("❓ Why does this matter?"):
            st.markdown("Claude (the AI) read the raw statistics and wrote this analysis for you. It highlights what's important, what to watch out for, and what it recommends — all in plain English.")
        insight = eda.get("claude_insight", "")
        if insight:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)


    # ── TAB 2: FEATURES ────────────────────────────────────────────────────────
    with tab2:
        feat = state.get("feature_report", {})
        if feat:
            st.markdown("### How your data was transformed")
            with st.expander("❓ Why does data need to be transformed?"):
                st.markdown("""
                Machine learning models can't work with raw messy data. They need:
                - **No blank values** — the agent fills these in intelligently
                - **Numbers only** — text categories (like "Male"/"Female") need to be converted to numbers
                - **Similar scales** — if one column goes 1-10 and another goes 1,000,000-2,000,000, the model gets confused. Scaling fixes this.

                The Feature Agent did all of this automatically and explains every decision below.
                """)

            final_shape = feat.get("final_shape", [0, 0])
            c1, c2, c3 = st.columns(3)
            c1.metric("Features used for training", len(feat.get("final_features", [])))
            c2.metric("Training rows", f"{final_shape[0]:,}")
            c3.metric("Transformations applied", sum([
                len(feat.get("missing_handling", [])),
                len(feat.get("encoding", [])),
                len(feat.get("scaling", [])),
                len(feat.get("variance_removal", [])),
            ]))

            col_a, col_b = st.columns(2)
            with col_a:
                if feat.get("missing_handling"):
                    st.markdown("**🩹 Missing values — how they were filled:**")
                    for s in feat["missing_handling"]:
                        st.markdown(f"- {s}")
                if feat.get("variance_removal"):
                    st.markdown("**🗑️ Removed (not useful):**")
                    for s in feat["variance_removal"]:
                        st.markdown(f"- {s}")
            with col_b:
                if feat.get("encoding"):
                    st.markdown("**🔢 Text → Numbers:**")
                    for s in feat["encoding"]:
                        st.markdown(f"- {s}")
                if feat.get("scaling"):
                    st.markdown("**📏 Scaling applied:**")
                    for s in feat["scaling"]:
                        st.markdown(f"- {s}")

            st.markdown("#### 🧠 Why these choices were made")
            insight = feat.get("claude_insight", "")
            if insight:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)


    # ── TAB 3: MODELS ─────────────────────────────────────────────────────────
    with tab3:
        model_r = state.get("model_report", {})
        if model_r:
            best = model_r.get("best_model", {})
            best_name = best.get("name", "Unknown")
            results = model_r.get("all_results", [])

            st.markdown("### 🏆 The winning model")
            with st.expander("❓ What is a model? How do we pick the best one?"):
                st.markdown("""
                A **machine learning model** is an algorithm that learns patterns from your data and uses them to make predictions on new data.

                AutoAgent tested **5 different model types** — think of them as 5 different "strategies" for solving your prediction problem. Each one has different strengths.

                **How we picked the winner:** Each model was tested on data it had never seen before. The one with the highest score on that unseen data wins. We also used **cross-validation** — testing the model 5 different ways to make sure the score is reliable, not just lucky.
                """)

            # Winner announcement
            if task == "classification":
                score_key = "cv_f1_mean"
                score_label = "F1 Score"
                score_explain = "F1 Score measures how well the model predicts both categories. 1.0 = perfect. 0.5 = random guessing."
            else:
                score_key = "cv_r2_mean"
                score_label = "R² Score"
                score_explain = "R² measures how well the model explains the variation in the target. 1.0 = perfect. 0.0 = model is useless."

            score_val = best.get(score_key, 0)
            score_pct = f"{score_val:.1%}"

            bx1, bx2, bx3 = st.columns(3)
            with bx1:
                st.markdown(f'<div class="big-result">🏆 {best_name}</div><div class="result-label">Winning model</div>', unsafe_allow_html=True)
            with bx2:
                st.markdown(f'<div class="big-result">{score_pct}</div><div class="result-label">{score_label} (cross-validated)</div>', unsafe_allow_html=True)
            with bx3:
                quality = "Excellent 🟢" if score_val > 0.85 else ("Good 🟡" if score_val > 0.70 else "Fair 🟠")
                st.markdown(f'<div class="big-result">{quality}</div><div class="result-label">Model quality</div>', unsafe_allow_html=True)

            st.caption(f"ℹ️ {score_explain}")

            st.markdown("---")
            st.markdown("#### All 5 models compared")

            if results:
                df_r = pd.DataFrame(results)
                colors = ["#22c55e" if r["name"] == best_name else "#94a3b8" for r in results]
                if score_key in df_r.columns:
                    fig = go.Figure(go.Bar(
                        x=df_r["name"], y=df_r[score_key],
                        marker_color=colors,
                        text=[f"{v:.1%}" for v in df_r[score_key]],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title=f"{score_label} by model (green = winner)",
                        height=350, yaxis_title=score_label,
                        plot_bgcolor="white", paper_bgcolor="white",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Detailed scores:**")
                st.dataframe(df_r.style.highlight_max(axis=0, color="#dcfce7"), use_container_width=True, hide_index=True)

            # SHAP
            shap_imp = model_r.get("shap_importance", {})
            if shap_imp:
                st.markdown("---")
                st.markdown("#### 🔍 What drove the predictions?")
                with st.expander("❓ What is SHAP and why does it matter?"):
                    st.markdown("""
                    **SHAP** (SHapley Additive exPlanations) is a technique that opens the "black box" of AI models.

                    It answers: *"Which columns in my data were most important for making predictions?"*

                    A higher SHAP value = that column had more influence on predictions. This is called **explainable AI** — you can see exactly why the model made its decisions, not just what it predicted.
                    """)

                df_shap = pd.DataFrame(list(shap_imp.items()), columns=["Feature", "Importance"]).sort_values("Importance")
                fig = px.bar(df_shap, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="Blues",
                             title="Feature importance — what drives your predictions")
                fig.update_layout(height=max(350, len(shap_imp) * 28), plot_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)

                top3 = list(shap_imp.items())[:3]
                st.markdown("**Top 3 most important predictors:**")
                for i, (f, v) in enumerate(top3):
                    st.markdown(f"{i+1}. **`{f}`** — importance score: `{v}`")

            st.markdown("---")
            st.markdown("#### 🧠 Claude's full recommendation")
            with st.expander("❓ What should I do with this?"):
                st.markdown("This is Claude's full written analysis — it explains which model won and why, what the scores mean in practice, the most important features, and what you should do next.")
            insight = model_r.get("claude_insight", "")
            if insight:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)


    # ── NEXT STEPS ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ✅ What can you do now?")
    col_n1, col_n2, col_n3 = st.columns(3)
    with col_n1:
        st.markdown("**📤 Share your results**\nTake screenshots of the charts and Claude's analysis. These are ready to present or include in a report.")
    with col_n2:
        st.markdown("**🔁 Try with your own data**\nNow that you've seen it work, upload your own dataset and predict something meaningful to you.")
    with col_n3:
        st.markdown("**🚀 Extend AutoAgent**\nThe next agents to build: Optimizer (tune the model further), Deploy (auto-generate an API), Monitor (track model drift over time).")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("AutoAgent · Built with LangGraph + Claude API by Pavan · github.com/pavan190602/autoagent · For questions, open a GitHub issue.")
