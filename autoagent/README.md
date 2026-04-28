# 🤖 AutoAgent

### The AI that answers questions from your spreadsheet data — no experience needed.

---

## The simplest way to understand what this does

You have a spreadsheet. You have a question.

**AutoAgent gives you the answer.**

---

**Here are some real examples:**

| You have... | You want to know... | AutoAgent tells you... |
|---|---|---|
| A list of customers | Which ones will stop paying? | "Customer ID 4821 has 87% chance of leaving" |
| A list of houses | What price should I list this one for? | "Based on size and location: $342,000" |
| A list of patients | Who is at risk of being readmitted? | "Patients over 65 with diabetes have 3x higher risk" |
| A list of loan applications | Who is likely to default? | "Applications with debt ratio above 0.4 are high risk" |

**That's it. That's what AutoAgent does.**

---

## Do I need to know anything about AI or coding?

**No.**

The only things you need:
- A spreadsheet (Excel or CSV file)
- A free account on Anthropic's website
- 5 minutes to set it up

After that, you upload your file, click a button, and read the results.

---

## What actually happens when I click "Run"?

AutoAgent sends three "agents" (think: AI workers) to do different jobs on your data.

**🔍 Agent 1 reads your spreadsheet**

It goes through every column and row. It finds things like:
- Some rows have blank cells — it flags them
- Some numbers are weirdly high or low compared to the rest — it flags those too
- It figures out which columns seem most related to what you're trying to predict

Then it writes a summary for you in plain English. Like a report from a real analyst.

---

**⚙️ Agent 2 prepares the data for AI**

AI can't work with messy data. So this agent:
- Fills in the blank cells intelligently (e.g. uses the average for that column)
- Converts text like "Yes/No" or "Male/Female" into numbers that AI understands
- Makes sure all the numbers are on similar scales so no single column dominates

It explains every single decision it makes, so you know exactly what was done.

---

**🤖 Agent 3 finds the best prediction method**

This agent tries 5 different AI approaches on your data — like asking 5 different experts and seeing who gives the best answer. It then:
- Picks the best performing one
- Shows you a score (how accurate it is)
- Shows you WHICH columns in your data were most important for making the prediction
- Writes a recommendation explaining the result

---

## What do I get at the end?

Three tabs of results — all explained in plain English:

**Tab 1 — "What's in my data?"**
A summary of your spreadsheet: how many rows, which columns had blank values, which columns seem related to what you're predicting. Plus an AI-written paragraph summarising the key findings.

**Tab 2 — "How was it prepared?"**
A log of every change made to your data before training — what was filled in, what was converted, what was removed and why.

**Tab 3 — "Which method won?"**
A comparison of all 5 approaches tested. The winner is highlighted. You get:
- A score showing how accurate it is (explained in plain English)
- A chart showing which columns in your data matter most
- A written recommendation from the AI

---

## How do I know if my result is good?

AutoAgent uses plain language to tell you:

> 🟢 **Excellent** — The model correctly predicted the right answer more than 85% of the time on data it had never seen before.

> 🟡 **Good** — Correct more than 70% of the time. Useful, but there may be room to improve.

> 🟠 **Fair** — Getting there, but the data may need more rows or better columns to improve.

You don't need to interpret numbers. AutoAgent interprets them for you.

---

## Setup Guide — step by step from scratch

> ⏱️ Takes about 5–10 minutes. Do this once, then you can use AutoAgent any time.

---

### Step 1 — Install Python (the language AutoAgent is built with)

Think of Python as the engine under the hood. You need it installed but you never have to write any Python yourself.

**Check if you already have it:**

Open your terminal:
- **Mac:** Press `Cmd + Space`, type "Terminal", hit Enter
- **Windows:** Press `Win + R`, type "cmd", hit Enter
- **Linux:** You know where it is 😄

Then type this and press Enter:
```
python --version
```

If you see something like `Python 3.11.2` → you're good, skip to Step 2.

If you see an error or `Python 2.x` → go to **[python.org/downloads](https://python.org/downloads)** and download the latest version.

> ⚠️ **Windows users:** During the Python installation, there is a checkbox that says **"Add Python to PATH"**. Make sure it is checked before you click Install. This is easy to miss and causes problems later.

---

### Step 2 — Download AutoAgent

**If you have Git installed:**
```bash
git clone https://github.com/pavan190602/autoagent.git
cd autoagent
```

**If you don't know what Git is:**
1. Look at the top of this GitHub page
2. Click the green button that says **"Code"**
3. Click **"Download ZIP"**
4. Find the ZIP file in your Downloads folder and unzip it
5. Open your terminal and navigate to that folder:
   - **Mac/Linux:** `cd ~/Downloads/autoagent`
   - **Windows:** `cd C:\Users\YourName\Downloads\autoagent`

---

### Step 3 — Install the packages AutoAgent needs

In your terminal, in the autoagent folder, run:
```bash
pip install -r requirements.txt
```

This will install everything automatically. It takes 2–5 minutes. You'll see a lot of text scrolling — that's completely normal.

**If pip doesn't work on Mac, try:**
```bash
pip3 install -r requirements.txt
```

**If you get a permissions error on Mac/Linux:**
```bash
pip install -r requirements.txt --user
```

---

### Step 4 — Get your free API key

AutoAgent uses Claude (made by Anthropic) to write the plain-English explanations. To access Claude, you need a free key.

**Getting it takes 2 minutes:**

1. Open your browser and go to **[console.anthropic.com](https://console.anthropic.com)**
2. Click "Sign Up" — create a free account with your email
3. Once you're logged in, click **"API Keys"** in the left sidebar
4. Click **"Create Key"**
5. Give it any name (e.g. `my-autoagent-key`)
6. **Copy the key** — it looks like: `sk-ant-api03-AbCdEfGh...`

> 🔒 This key is like a password. Don't share it or put it on the internet.

> 💰 Cost: Anthropic gives you free credits when you sign up. Running AutoAgent on a typical dataset costs a few cents at most.

**Now save your key:**

In your terminal, inside the autoagent folder:

```bash
# Mac or Linux — paste this with your actual key:
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env

# Windows — paste this with your actual key:
echo ANTHROPIC_API_KEY=sk-ant-your-key-here > .env
```

Or do it manually:
1. Open the autoagent folder in your file explorer
2. Create a new text file called `.env` (just `.env` — no `.txt` at the end)
3. Inside it, write: `ANTHROPIC_API_KEY=sk-ant-your-key-here`
4. Save it

---

### Step 5 — Run AutoAgent

```bash
streamlit run app.py
```

Your browser will open automatically and show the AutoAgent interface.

If it doesn't open automatically, go to: **http://localhost:8501**

---

### Step 6 — Use it

**First time? Use a sample dataset:**

1. In the AutoAgent interface, find **Step 2 — Upload Data**
2. Click the **"Use a sample dataset"** tab
3. Select **"Titanic — predict who survived"**
4. Click **"Load this sample dataset"**
5. It will automatically fill in the target (what to predict) for you
6. Click the big **"Run AutoAgent"** button
7. Wait 1–3 minutes
8. Explore your results in the three tabs

**With your own data:**

1. Click **"Upload my own file"**
2. Upload your Excel or CSV file
3. Look at the preview — find the column name you want to predict
4. Select that column as your "target"
5. Choose whether your target is a **category** (Yes/No, names, labels) or a **number** (price, score, amount)
6. Click Run

---

## Something went wrong?

| What you see | What to do |
|---|---|
| `ModuleNotFoundError: No module named 'X'` | Run `pip install X` (replace X with the module name in the error) |
| `ANTHROPIC_API_KEY not configured` | Check your `.env` file is in the autoagent folder and contains your key |
| Browser opened but page is blank | Wait 15 seconds and refresh |
| Nothing opened in browser | Go to `http://localhost:8501` manually |
| `Port 8501 is already in use` | Run `streamlit run app.py --server.port 8502` then go to `http://localhost:8502` |
| `pip: command not found` on Mac | Try `pip3` instead |

**Still stuck?** [Open an issue on GitHub](https://github.com/pavan190602/autoagent/issues) — paste the exact error message and describe what step you were on.

---

## What kind of data works best?

✅ Works well with:
- At least 100 rows (500+ is better)
- A clear column you want to predict
- A mix of number columns and category columns

⚠️ May struggle with:
- Less than 50 rows — not enough data to learn from
- Free text columns like paragraphs or descriptions
- Dates (works, but less informative without feature engineering)

❌ Won't work with:
- Images or audio files
- Multiple spreadsheets that need to be joined together (merge them first)

---

## For developers

**Architecture:**
```
Streamlit UI → FastAPI Gateway → LangGraph Orchestrator
                                        ↓
                         EDA Agent → Feature Agent → Model Agent
                                        ↓
                                   Claude API (tool_use)
                                        ↓
                         sklearn · XGBoost · LightGBM · SHAP
```

**REST API:**
```bash
python api.py   # docs at localhost:8000/docs

curl -X POST http://localhost:8000/run \
  -F "file=@sample_data/churn.csv" \
  -F "problem_description=Predict churn" \
  -F "target_column=churn" \
  -F "task_type=classification"
```

**Tech stack:** LangGraph · Claude API (claude-sonnet-4, tool_use) · scikit-learn · XGBoost · LightGBM · SHAP · FastAPI · Streamlit · Plotly · Pandas

---

## Roadmap

- [ ] Optimizer Agent — automatically tune the model for higher accuracy
- [ ] Deploy Agent — generate a ready-to-use API from your trained model
- [ ] Monitor Agent — alert you when predictions start getting worse over time
- [ ] RAG knowledge base — smarter agent decisions using ML literature
- [ ] MLflow tracking — save and compare experiments

---

## Author

**Pavan** · MS Computer Science, University of Central Missouri  
AI/ML Engineering · LLMs · RAG · Agentic AI  
[GitHub](https://github.com/pavan190602) · [LinkedIn](https://linkedin.com/in/pavan)

---

*MIT License · Free to use and modify · Attribution appreciated*
