# 🤖 AutoAgent — Complete Beginner Setup Guide

> **Who this is for:** Someone who has never done machine learning, maybe hasn't coded much, and just wants to get this running. We assume nothing.

---

## What you'll be able to do at the end

Upload any spreadsheet → AutoAgent predicts outcomes from it → You get charts, explanations, and model comparisons — all explained in plain English.

---

## What you need before starting

| Requirement | What it is | Do you have it? |
|-------------|-----------|-----------------|
| A computer | Windows, Mac, or Linux | ✅ Presumably yes |
| Internet connection | To download things | ✅ Presumably yes |
| Python | A programming language (free) | Maybe — check below |
| An Anthropic API key | Lets AutoAgent use Claude AI (free tier available) | Probably not yet — covered in Step 2 |

---

## Step 1 — Install Python

Python is the programming language AutoAgent is built with. You need version 3.10 or newer.

**Check if you already have Python:**

Open your terminal (Mac/Linux) or Command Prompt (Windows) and type:
```
python --version
```

If you see `Python 3.10.x` or higher, skip to Step 2.

If you get an error or see Python 2.x, install Python:

1. Go to **python.org/downloads**
2. Click the big yellow "Download Python 3.x.x" button
3. Run the installer
4. ⚠️ **Windows users:** During installation, check the box that says **"Add Python to PATH"** — this is important!
5. After install, close and reopen your terminal, then run `python --version` again to confirm

---

## Step 2 — Get your Anthropic API Key

AutoAgent uses Claude (Anthropic's AI) to analyse your data and explain results. You need a key to access it.

1. Open your browser and go to **console.anthropic.com**
2. Click **"Sign Up"** and create a free account (just email + password)
3. Once logged in, look for **"API Keys"** in the left sidebar menu
4. Click **"Create Key"**
5. Give it a name like `autoagent` (any name is fine)
6. **Copy the key** — it looks like: `sk-ant-api03-xxxxxxxxxxxx`
7. Save it somewhere safe (like a notes app) — you'll need it in a moment

> ⚠️ **Important:** This key is like a password. Never share it or put it in a public GitHub repo.

> 💡 **Cost:** Anthropic gives you free credits to start. Running AutoAgent on a typical dataset costs a few cents of API credits.

---

## Step 3 — Download AutoAgent

**Option A — Download as ZIP (easiest, no Git needed):**

1. Go to `github.com/pavan190602/autoagent`
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Unzip the file somewhere on your computer (e.g. your Desktop or Documents)
5. Open your terminal/Command Prompt and navigate to that folder:

```bash
# Mac/Linux example:
cd ~/Desktop/autoagent

# Windows example:
cd C:\Users\YourName\Desktop\autoagent
```

**Option B — Git clone (if you have Git installed):**
```bash
git clone https://github.com/pavan190602/autoagent.git
cd autoagent
```

---

## Step 4 — Install the required packages

AutoAgent uses several Python packages (libraries). Install them all with one command:

```bash
pip install -r requirements.txt
```

This will take 2–5 minutes. You'll see a lot of text scrolling — that's normal.

**If you get a "pip not found" error on Mac/Linux, try:**
```bash
pip3 install -r requirements.txt
```

**If you get permission errors on Mac/Linux:**
```bash
pip install -r requirements.txt --user
```

> 💡 **What's being installed?** Things like scikit-learn (ML algorithms), XGBoost (a powerful model), LangGraph (agent framework), Streamlit (the web UI), and SHAP (explainability). You don't need to know what these are — they just need to be installed.

---

## Step 5 — Set up your API key

Create a file called `.env` in the autoagent folder.

**Mac/Linux:**
```bash
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

**Windows (Command Prompt):**
```
echo ANTHROPIC_API_KEY=your-key-here > .env
```

Replace `your-key-here` with the actual key you copied in Step 2.

**Or do it manually:**
1. Open the `autoagent` folder in your file explorer
2. Create a new text file
3. Name it `.env` (with the dot at the start, no .txt extension)
4. Open it and type: `ANTHROPIC_API_KEY=sk-ant-api03-yourkeyhere`
5. Save it

---

## Step 6 — Run AutoAgent

```bash
streamlit run app.py
```

After a few seconds, your browser will automatically open to `http://localhost:8501` with the AutoAgent UI.

If the browser doesn't open automatically, open it yourself and go to: **http://localhost:8501**

> 💡 **To stop AutoAgent:** Go back to your terminal and press `Ctrl + C`

---

## Step 7 — Use AutoAgent

Once the browser is open, you'll see a step-by-step guided interface. Here's what to do:

### Option A — Try with a sample dataset first

1. In **Step 2** of the UI, click the **"Use a sample dataset"** tab
2. Select **"🚢 Titanic"** (it's the most classic ML dataset)
3. Click **"Load this sample dataset"**
4. In **Step 3**, it will auto-fill the target column (`Survived`) and task type (`classification`)
5. Optionally write a description: `"Predict whether a passenger survived the Titanic disaster"`
6. Click **"🚀 Run AutoAgent"**
7. Wait 1–3 minutes while the agents run
8. Explore your results in the 3 tabs below

### Option B — Use your own data

1. In **Step 2**, click **"Upload my own file"**
2. Upload a CSV or Excel file
3. Look at the preview — find the column name you want to predict
4. In **Step 3**, type that column name in the "Target column" box
5. Choose Classification or Regression (see the guide in the expandable section)
6. Click Run

---

## Common problems & solutions

### "ModuleNotFoundError: No module named 'X'"
You're missing a package. Run:
```bash
pip install X
```
(Replace X with whatever name appeared in the error)

### "ANTHROPIC_API_KEY not found" or "API key invalid"
- Make sure your `.env` file exists in the autoagent folder
- Make sure the key starts with `sk-ant-`
- Try entering the key directly in the UI sidebar instead

### "Port 8501 is already in use"
Another streamlit app is running. Either close it, or run:
```bash
streamlit run app.py --server.port 8502
```
Then go to `http://localhost:8502` in your browser

### The browser opened but the page won't load
Wait 10-15 seconds and refresh. The first load can be slow.

### "pip is not recognised" on Windows
Try `py -m pip install -r requirements.txt` instead

### Installation is taking forever
Normal — especially XGBoost and scikit-learn are large. Let it finish.

---

## What the results mean

### EDA (Exploratory Data Analysis) tab
Shows you what's in your data — how many rows, missing values, which columns are most related to what you're predicting.

**Plain English:** "Here's what I found interesting in your spreadsheet."

### Feature Engineering tab
Shows you how the data was cleaned and transformed before training.

**Plain English:** "Here's how I prepared your data for the AI."

### Models tab
Shows you the 5 models that were tested and which one won, with a score.

**What the scores mean:**
- **F1 Score** (for classification): 0% = useless, 100% = perfect. Above 80% is very good.
- **R² Score** (for regression): 0 = useless, 1.0 = perfect. Above 0.7 is good.

**SHAP chart:** Shows which columns in your data had the most influence on predictions. Higher bar = more important.

---

## Frequently asked questions

**Q: Is my data safe? Does it get sent anywhere?**
A: Your data is processed only on your computer. The only thing sent externally is a statistical summary (not your actual rows) to the Claude API for generating the written explanations.

**Q: How much does the API cost?**
A: Running AutoAgent on a typical 500-row dataset costs approximately $0.01–0.05 USD in API credits. Anthropic gives you free credits when you sign up.

**Q: Can I use any CSV file?**
A: Yes, as long as it has at least 50–100 rows and the column you want to predict is actually in the file.

**Q: What if my target column has text values like "Yes"/"No"?**
A: That's fine — choose "Classification" as the task type and AutoAgent will handle it automatically.

**Q: Can I run this on Google Colab or a cloud server?**
A: Yes. Install requirements with `pip install -r requirements.txt` and run `streamlit run app.py`. Use ngrok or Colab's port forwarding to access the UI.

---

## Project structure (for the curious)

```
autoagent/
├── app.py                   ← The beginner-friendly UI (run this)
├── streamlit_app.py         ← Advanced UI with more technical detail
├── api.py                   ← REST API (for developers)
├── orchestrator.py          ← The LangGraph pipeline that connects agents
├── agents/
│   ├── eda_agent.py         ← Agent 1: explores and profiles your data
│   ├── feature_agent.py     ← Agent 2: cleans and transforms your data
│   └── model_agent.py       ← Agent 3: trains models and picks the best
├── sample_data/
│   └── churn.csv            ← Sample dataset to test with
├── requirements.txt         ← List of Python packages needed
└── .env                     ← Your API key (you create this)
```

---

## Need help?

- Open an issue on GitHub: `github.com/pavan190602/autoagent/issues`
- Describe what you did, what you expected, and what error you got

---

*AutoAgent · Built by Pavan · MS Computer Science, University of Central Missouri*
