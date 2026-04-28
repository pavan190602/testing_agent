"""
Generate sample datasets for testing AutoAgent.
Run: python sample_data/generate.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

df = pd.DataFrame({
    "age":            np.random.randint(18, 70, n),
    "tenure_months":  np.random.randint(1, 72, n),
    "monthly_charges": np.round(np.random.uniform(20, 120, n), 2),
    "num_products":   np.random.randint(1, 5, n),
    "support_calls":  np.random.randint(0, 10, n),
    "contract_type":  np.random.choice(["Monthly", "Annual", "Two-Year"], n),
    "payment_method": np.random.choice(["Credit Card", "Bank Transfer", "E-Check"], n),
    "internet_service": np.random.choice(["DSL", "Fiber", "None"], n),
})

# Introduce missing values realistically
df.loc[np.random.choice(n, 30, replace=False), "monthly_charges"] = np.nan
df.loc[np.random.choice(n, 20, replace=False), "support_calls"] = np.nan

# Target — churn probability influenced by features
churn_prob = (
    0.05
    + 0.003 * df["support_calls"].fillna(3)
    - 0.002 * df["tenure_months"]
    + 0.001 * df["monthly_charges"].fillna(60)
    + (df["contract_type"] == "Monthly").astype(float) * 0.15
)
churn_prob = churn_prob.clip(0.02, 0.85)
df["churn"] = (np.random.rand(n) < churn_prob).astype(int)

df.to_csv("sample_data/churn.csv", index=False)
print(f"Generated churn.csv: {df.shape}")
print(df.head())
print(f"\nChurn rate: {df['churn'].mean():.2%}")
