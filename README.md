# 🤖 AutoAgent — Autonomous ML Engineering System

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-purple)](https://github.com/langchain-ai/langgraph)
[![Claude](https://img.shields.io/badge/Claude-Anthropic-orange)](https://anthropic.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What is this?

You have a spreadsheet with historical data.
AutoAgent studies that data and **predicts answers for new rows it has never seen.**

No coding. No machine learning experience. Just upload your file and click Run.

---

## Real world examples

### 🏦 Bank — Will this person repay their loan?
**You have:** 10,000 past loan applications with columns like Age, Income, Credit Score, Loan Amount
**Answer column:** `paid_back` (Yes / No)
**AutoAgent learns:** Who paid back loans and who didn't
**Then predicts:** For a NEW applicant — will they pay it back?

---

### 🏠 Real Estate — What is this house worth?
**You have:** 5,000 houses that already sold with columns like Size, Bedrooms, Location, Age
**Answer column:** `sale_price`
**AutoAgent learns:** What makes houses sell for more or less
**Then predicts:** What price should THIS new house be listed at?

---

### 🛒 Online Store — Which customers are about to leave?
**You have:** 50,000 customers with columns like Purchase Frequency, Last Order Date, Total Spent
**Answer column:** `churned` (Yes / No)
**AutoAgent learns:** What kind of customers stop buying
**Then predicts:** Which of your CURRENT customers are about to leave?

---

### 🏥 Hospital — Which patients will be readmitted?
**You have:** 20,000 past patients with columns like Age, Diagnosis, Medications, Length of Stay
**Answer column:** `readmitted` (Yes / No)
**AutoAgent learns:** Which patients tend to come back within 30 days
**Then predicts:** For patients being discharged TODAY — who is at risk?

---

### ⚽ Sports — Who will win this match?
**You have:** 3,000 past matches with columns like Home Team, Away Team, Recent Form, Injuries
**Answer column:** `result` (Win / Draw / Loss)
**AutoAgent learns:** What factors lead to wins
**Then predicts:** Who will win THIS weekend's match?

---

### 🚗 Car Dealership — What is this used car worth?
**You have:** 8,000 cars that were sold with columns like Brand, Year, Mileage, Condition
**Answer column:** `selling_price`
**AutoAgent learns:** What makes a car worth more or less
**Then predicts:** What is THIS used car worth today?

---

### 👩‍💼 HR — Which employees are about to quit?
**You have:** 2,000 past employees with columns like Salary, Years at Company, Manager Rating
**Answer column:** `resigned` (Yes / No)
**AutoAgent learns:** What makes employees leave
**Then predicts:** Which of your CURRENT employees are most likely to quit?

---

## The pattern is always the same

```
Old data with known answers  →  AutoAgent learns  →  Predicts for new data
```

**Your spreadsheet is the old data with known answers. That is the input. Every time.**

---

## What you give AutoAgent

Just three things:

| What | Example |
|------|---------|
| **A spreadsheet file** (CSV or Excel) | `customers.csv` |
| **Which column to predict** | `churned` |
| **Is the answer a category or a number?** | Category (Yes/No) → Classification. Number (price) → Regression |

---

## What AutoAgent gives you back

Three sections of results:

**🔍 Section 1 — "What's in my data?"**
Charts and a written summary of your spreadsheet. Which columns have blank values. Which columns are most related to what you're predicting.

**⚙️ Section 2 — "How was it prepared?"**
Every change made to your data before training — explained in plain English. What was filled in, what was converted, what was removed and why.

**🤖 Section 3 — "Which method won?"**
AutoAgent tests 5 different prediction approaches and picks the best one. You get:
- A score showing how accurate it is
- A chart showing which columns in your data matter most for predictions
- A written recommendation from Claude AI

---

## How accurate is it? What do the scores mean?

| Score | What it means |
|-------|--------------|
| 🟢 Above 85% | Excellent — gets it right most of the time |
| 🟡 70–85% | Good — useful for most real decisions |
| 🟠 Below 70% | Fair — may need more data or better columns |

AutoAgent tells you which bracket you're in, in plain English. You don't have to interpret numbers yourself.

---

## Setup — from scratch to running in 10 minutes

### 1. Install Python

Open your terminal (Mac: `Cmd+Space` → type Terminal. Windows: `Win+R` → type cmd) and run:

```bash
python --version
```

Need 3.10 or higher. Don't have it? → [python.org/downloads](https://python.org/downloads)

> ⚠️ Windows users: During install, check **"Add Python to PATH"** before clicking Install.

---

### 2. Download AutoAgent

```bash
git clone https://github.com/pavan190602/autoagent.git
cd autoagent
```

No Git? Click the green **"Code"** button on this page → **Download ZIP** → unzip it → open that folder in your terminal.

---

### 3. Install everything AutoAgent needs

```bash
pip install -r requirements.txt
```

Takes 2–5 minutes. Lots of text will scroll — that's normal.

> Mac/Linux if pip doesn't work: try `pip3`
> Permission error: add `--user` to the command

---

### 4. Get your free API key

AutoAgent uses Claude AI (by Anthropic) to write the plain-English explanations. You need a free key:

1. Go to **[console.anthropic.com](https://console.anthropic.com)**
2. Sign up for free
3. Click **API Keys** → **Create Key**
4. Copy it — looks like `sk-ant-api03-...`

Save it in the autoagent folder:

```bash
# Mac/Linux:
echo "ANTHROPIC_API_KEY=your-key-here" > .env

# Windows:
echo ANTHROPIC_API_KEY=your-key-here > .env
```

---

### 5. Run it

```bash
streamlit run app.py
```

Browser opens at **http://localhost:8501** automatically.

---

### 6. Try it with sample data first

If you don't have your own dataset yet:

1. In the UI, go to **Step 2 → "Use a sample dataset"** tab
2. Pick **Titanic — predict who survived**
3. Click **Load sample dataset**
4. Everything auto-fills
5. Click **Run AutoAgent**
6. Wait 1–3 minutes
7. Explore the three result tabs

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `API key not found` | Check your `.env` file is in the autoagent folder |
| Browser didn't open | Go to `http://localhost:8501` manually |
| Port already in use | `streamlit run app.py --server.port 8502` |
| `pip not found` on Mac | Use `pip3` instead |

Still stuck? [Open an issue](https://github.com/pavan190602/autoagent/issues) and paste the error message.

---

## How it works inside (for technical readers)

```
Streamlit UI (app.py)
        ↓
LangGraph Orchestrator
        ↓
EDA Agent → Feature Agent → Model Agent
        ↓              ↓             ↓
   Claude API     Claude API    Claude API
   (insights)   (explanation)  (recommendation)
        ↓
sklearn · XGBoost · LightGBM · SHAP
```

Each agent uses Claude's **tool_use API** — Claude actually reads the real data statistics, not a generic prompt. This is why the explanations are specific to your dataset, not generic advice.

**Tech stack:** LangGraph · Claude API (claude-sonnet-4) · scikit-learn · XGBoost · LightGBM · SHAP · FastAPI · Streamlit · Plotly · Pandas · NumPy

---

## REST API (for developers)

```bash
python api.py
# Docs: http://localhost:8000/docs
```

```bash
curl -X POST http://localhost:8000/run \
  -F "file=@sample_data/churn.csv" \
  -F "problem_description=Predict customer churn" \
  -F "target_column=churn" \
  -F "task_type=classification"
```

---

## Project files

```
autoagent/
├── app.py               ← The UI — run this
├── orchestrator.py      ← LangGraph pipeline
├── agents/
│   ├── eda_agent.py     ← Agent 1: reads and analyses data
│   ├── feature_agent.py ← Agent 2: cleans and prepares data
│   └── model_agent.py   ← Agent 3: trains models, picks best, explains
├── api.py               ← REST API for developers
├── sample_data/
│   └── churn.csv        ← Sample dataset to test with
├── requirements.txt     ← All packages needed
└── .env                 ← Your API key (you create this)
```

---

## Roadmap

- [ ] Optimizer Agent — automatically tune the model for higher accuracy
- [ ] Deploy Agent — generate a working API from your trained model
- [ ] Monitor Agent — alert when predictions degrade over time
- [ ] MLflow tracking — save and compare experiments

---

## Author

**Pavan** · MS Computer Science, University of Central Missouri
AI/ML Engineering · LLMs · RAG · Agentic AI
[GitHub](https://github.com/pavan190602) · [LinkedIn](https://linkedin.com/in/pavan)

---

*MIT License · Free to use and modify*
