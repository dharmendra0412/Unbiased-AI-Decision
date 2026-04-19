"""
generate_sample_data.py
Generates a realistic, intentionally biased hiring dataset for demo purposes.
Run this once to create 'sample_hiring_data.csv'.
"""
import pandas as pd
import numpy as np

np.random.seed(42)
N = 500

genders   = np.random.choice(["Male", "Female"], size=N, p=[0.55, 0.45])
races     = np.random.choice(["White", "Black", "Asian", "Hispanic"], size=N, p=[0.50, 0.20, 0.20, 0.10])
ages      = np.random.randint(22, 55, size=N)
exp_years = np.clip(np.random.normal(5, 3, N).astype(int), 0, 20)
edu_level = np.random.choice(["High School", "Bachelor's", "Master's", "PhD"],
                               size=N, p=[0.15, 0.50, 0.25, 0.10])
coding_score = np.clip(np.random.normal(70, 15, N), 10, 100).astype(int)

# ---- Introduce systemic bias ----
# Male candidates: higher base hire probability
# White/Asian candidates: slight boost
hire_prob = 0.35 * np.ones(N)
hire_prob[genders == "Male"]    += 0.25   # strong gender bias
hire_prob[races  == "White"]    += 0.10   # racial bias
hire_prob[races  == "Asian"]    += 0.05
hire_prob[exp_years >= 5]       += 0.10   # legitimate signal
hire_prob[coding_score >= 75]   += 0.10   # legitimate signal
hire_prob[edu_level == "Master's"] += 0.05
hire_prob[edu_level == "PhD"]      += 0.08
hire_prob = np.clip(hire_prob, 0, 1)

hired = (np.random.rand(N) < hire_prob).astype(int)

df = pd.DataFrame({
    "Age":             ages,
    "Gender":          genders,
    "Race":            races,
    "Years_Experience": exp_years,
    "Education":       edu_level,
    "Coding_Score":    coding_score,
    "Hired":           hired,
})

df.to_csv("sample_hiring_data.csv", index=False)
print(f"✅  Generated sample_hiring_data.csv  ({N} rows)")
print(df["Hired"].value_counts())
print("\nHire rate by Gender:\n", df.groupby("Gender")["Hired"].mean().round(3))
print("\nHire rate by Race:\n",   df.groupby("Race")["Hired"].mean().round(3))
