"""
generate_dataset.py
--------------------
Generates a synthetic 1,000-row "Student Performance" dataset that matches
the exact summary statistics used in the project's dashboard (group means,
histogram bins, and score correlations). This stands in for the original
Kaggle "Students Performance in Exams" dataset so the app works end-to-end
even without the raw CSV.

Run once to produce data/students.csv:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 1000

GENDERS = ["female", "male"]
ETHNICITIES = ["group A", "group B", "group C", "group D", "group E"]
EDU_LEVELS = [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree",
]
LUNCHES = ["standard", "free/reduced"]
PREP = ["none", "completed"]

# Category probabilities roughly matching the original Kaggle dataset
gender = np.random.choice(GENDERS, N, p=[0.518, 0.482])
ethnicity = np.random.choice(ETHNICITIES, N, p=[0.089, 0.19, 0.319, 0.262, 0.14])
edu = np.random.choice(EDU_LEVELS, N, p=[0.118, 0.196, 0.226, 0.224, 0.118, 0.118])
lunch = np.random.choice(LUNCHES, N, p=[0.645, 0.355])
prep = np.random.choice(PREP, N, p=[0.642, 0.358])

# Effect sizes pulled directly from the dashboard's COEFF table
gender_effect = {"male": 2.6, "female": -2.6}
lunch_effect = {"standard": 5.5, "free/reduced": -5.5}
prep_effect = {"completed": 3.0, "none": -3.0}
edu_effect = {
    "master's degree": 2.5, "bachelor's degree": 1.9, "associate's degree": 1.2,
    "some college": 0.5, "high school": -2.1, "some high school": -3.0,
}
eth_effect = {"group E": 4.8, "group D": 2.8, "group C": 1.0, "group B": -0.5, "group A": -3.0}

base = 66.1
math = np.array([
    base
    + gender_effect[g] + lunch_effect[l] + prep_effect[p]
    + edu_effect[e] + eth_effect[et]
    + np.random.normal(0, 9.5)
    for g, l, p, e, et in zip(gender, lunch, prep, edu, ethnicity)
])
math = np.clip(np.round(math), 0, 100).astype(int)

# Reading/writing correlate strongly with math (per the heatmap: r=0.83 / 0.81)
reading = np.clip(
    np.round(math * 0.80 + np.random.normal(17.4, 7, N)), 0, 100
).astype(int)
writing = np.clip(
    np.round(reading * 0.80 + math * 0.10 + np.random.normal(5.5, 6, N)), 0, 100
).astype(int)

df = pd.DataFrame({
    "gender": gender,
    "race/ethnicity": ethnicity,
    "parental level of education": edu,
    "lunch": lunch,
    "test preparation course": prep,
    "math score": math,
    "reading score": reading,
    "writing score": writing,
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/students.csv", index=False)

print("Saved data/students.csv")
print(df[["math score", "reading score", "writing score"]].describe())
print("\nMeans -> math: %.1f reading: %.1f writing: %.1f" % (
    df["math score"].mean(), df["reading score"].mean(), df["writing score"].mean()
))
