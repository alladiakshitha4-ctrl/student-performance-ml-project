"""
app.py — Student Performance ML Dashboard
==========================================
A Streamlit dashboard for the Student Performance ML project.

Pages:
  - Overview        : dataset summary + score distributions
  - Dataset Explorer : searchable / filterable data table
  - EDA / Charts      : breakdowns by education, ethnicity, prep course, correlation heatmap
  - Model Training    : trains & compares 8 regression models live with scikit-learn
  - Predict Score     : interactive form that predicts a student's math score

Run locally:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance ML",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

ORANGE = "#f97316"
ORANGE2 = "#fb923c"
BLUE = "#3b82f6"
GREEN = "#10b981"
YELLOW = "#eab308"
RED = "#ef4444"
BG = "#0f172a"
SURFACE = "#1e293b"
MUTED = "#94a3b8"
TEXT = "#e2e8f0"
BORDER = "#334155"

# ──────────────────────────────────────────────────────────────────────────
# THEME / CSS — dark dashboard look matching the original design
# ──────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
.stApp {{ background-color: {BG}; color: {TEXT}; }}
section[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}
h1, h2, h3, h4 {{ color: #f1f5f9 !important; }}
.metric-card {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 1.1rem 1.25rem; position: relative; overflow: hidden; margin-bottom: 0.5rem;
}}
.metric-card::before {{
    content: ''; position: absolute; bottom: 0; left: 0; right: 0;
    height: 3px; background: {ORANGE};
}}
.metric-label {{ font-size: 11px; color: {MUTED}; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 6px; }}
.metric-val {{ font-size: 28px; font-weight: 700; color: {ORANGE}; line-height: 1; }}
.metric-sub {{ font-size: 11px; color: #475569; margin-top: 4px; }}
.sec-label {{
    font-size: 15px; font-weight: 600; color: {TEXT};
    border-left: 3px solid {ORANGE}; padding-left: 10px; margin: 1.5rem 0 1rem;
}}
.result-card {{
    background: {SURFACE}; border: 2px solid {ORANGE}; border-radius: 16px;
    padding: 2rem; text-align: center;
}}
.result-score {{ font-size: 64px; font-weight: 800; line-height: 1; color: {ORANGE}; }}
.result-grade {{ font-size: 16px; font-weight: 600; margin-top: 8px; }}
.badge-best {{
    display: inline-block; font-size: 10px; font-weight: 700; padding: 2px 7px;
    border-radius: 4px; background: {ORANGE}; color: #fff; margin-left: 6px;
}}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "students.csv")


@st.cache_data
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run `python generate_dataset.py` first, "
            "or replace data/students.csv with your own Kaggle dataset "
            "(same column names)."
        )
    df = pd.read_csv(DATA_PATH)
    df["average score"] = df[["math score", "reading score", "writing score"]].mean(axis=1).round(1)
    return df


df = load_data()

FEATURES = ["gender", "race/ethnicity", "parental level of education", "lunch",
            "test preparation course", "reading score", "writing score"]
TARGET = "math score"
CAT_FEATURES = ["gender", "race/ethnicity", "parental level of education", "lunch",
                 "test preparation course"]
NUM_FEATURES = ["reading score", "writing score"]


@st.cache_resource
def train_models(data: pd.DataFrame):
    """Train all 8 regression models once and cache the results."""
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])

    model_defs = {
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200),
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "Lasso": Lasso(random_state=42),
        "K-Neighbors": KNeighborsRegressor(n_neighbors=7),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=6),
    }

    results = []
    fitted_pipelines = {}
    for name, est in model_defs.items():
        pipe = Pipeline([("prep", preprocessor), ("model", est)])
        pipe.fit(X_train, y_train)
        train_pred = pipe.predict(X_train)
        test_pred = pipe.predict(X_test)
        results.append({
            "Model": name,
            "Train R²": r2_score(y_train, train_pred),
            "Test R²": r2_score(y_test, test_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
            "MAE": mean_absolute_error(y_test, test_pred),
        })
        fitted_pipelines[name] = pipe

    results_df = pd.DataFrame(results).sort_values("Test R²", ascending=False).reset_index(drop=True)
    return results_df, fitted_pipelines


with st.spinner("Training models…"):
    results_df, pipelines = train_models(df)

BEST_MODEL_NAME = results_df.iloc[0]["Model"]

# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""<div style="padding:0.5rem 0 1rem;border-bottom:1px solid {BORDER};margin-bottom:0.5rem;">
        <div style="font-size:16px;font-weight:700;color:{ORANGE};">🎓 Student Performance</div>
        <div style="font-size:11px;color:#475569;margin-top:3px;">ML Project — Internship Studio</div>
        </div>""",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        ["📊 Overview", "🗂️ Dataset Explorer", "📈 EDA / Charts", "🤖 Model Training", "⚡ Predict Score"],
        label_visibility="collapsed",
    )
    st.markdown(
        f"""<div style="position:fixed;bottom:1rem;font-size:11px;color:#475569;">
        Dataset: Kaggle-style · {len(df):,} students · 8 features</div>""",
        unsafe_allow_html=True,
    )


def metric_card(label, value, sub):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-val">{value}</div>
        <div class="metric-sub">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def section_label(text):
    st.markdown(f'<div class="sec-label">{text}</div>', unsafe_allow_html=True)


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color=MUTED,
    margin=dict(l=10, r=10, t=10, b=10),
)

# ──────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ──────────────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown("## Student Performance Overview")
    st.caption("How gender, ethnicity, parental education, lunch & test prep affect exam scores")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Students", f"{len(df):,}", "dataset rows")
    with c2:
        metric_card("Avg Math Score", f"{df['math score'].mean():.1f}", "out of 100")
    with c3:
        metric_card("Avg Reading Score", f"{df['reading score'].mean():.1f}", "out of 100")
    with c4:
        metric_card("Avg Writing Score", f"{df['writing score'].mean():.1f}", "out of 100")

    section_label("Score distributions across all students")
    h1, h2, h3 = st.columns(3)
    for col, score_col, color, label in zip(
        [h1, h2, h3],
        ["math score", "reading score", "writing score"],
        [ORANGE, BLUE, GREEN],
        ["Math", "Reading", "Writing"],
    ):
        with col:
            st.caption(f"{label} Score Distribution")
            fig = px.histogram(df, x=score_col, nbins=10, color_discrete_sequence=[color])
            fig.update_layout(**PLOTLY_LAYOUT, height=220, bargap=0.1, showlegend=False)
            fig.update_xaxes(title=None, gridcolor=BORDER)
            fig.update_yaxes(title=None, gridcolor=BORDER)
            st.plotly_chart(fig, use_container_width=True, key=f"hist_{score_col}")

    section_label("Key findings")
    g1, g2 = st.columns(2)
    with g1:
        st.caption("Average Math Score by Gender")
        gender_avg = df.groupby("gender")["math score"].mean().reindex(["female", "male"])
        fig = px.bar(
            x=[g.capitalize() for g in gender_avg.index], y=gender_avg.values,
            color=[g.capitalize() for g in gender_avg.index],
            color_discrete_map={"Female": BLUE, "Male": ORANGE},
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=False)
        fig.update_xaxes(title=None, gridcolor=BORDER)
        fig.update_yaxes(title=None, gridcolor=BORDER)
        st.plotly_chart(fig, use_container_width=True, key="gender_chart")
    with g2:
        st.caption("Standard vs Free/Reduced Lunch")
        lunch_avg = df.groupby("lunch")["math score"].mean().reindex(["standard", "free/reduced"])
        fig = px.bar(
            x=["Standard", "Free / Reduced"], y=lunch_avg.values,
            color=["Standard", "Free / Reduced"],
            color_discrete_map={"Standard": GREEN, "Free / Reduced": RED},
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=False)
        fig.update_xaxes(title=None, gridcolor=BORDER)
        fig.update_yaxes(title=None, gridcolor=BORDER)
        st.plotly_chart(fig, use_container_width=True, key="lunch_chart")

    section_label("Project lifecycle")
    lifecycle = pd.DataFrame({
        "#": [1, 2, 3, 4, 5, 6, 7],
        "Phase": ["Data Collection", "Data Cleaning", "EDA", "Feature Engineering",
                  "Model Training", "Model Evaluation", "Best Model"],
        "Description": [
            "Kaggle Students Performance dataset", "Checked nulls & duplicates (0 found)",
            "Score distributions, group comparisons, correlations",
            "One-hot encoding categorical features",
            "8 sklearn regression models trained",
            "Compared on R², RMSE, MAE (80/20 split)",
            f"{BEST_MODEL_NAME} — R² {results_df.iloc[0]['Test R²']:.3f}",
        ],
        "Status": ["Done", "Done", "Done", "Done", "Done", "Done", "Best"],
    })
    st.dataframe(lifecycle, hide_index=True, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# PAGE: DATASET EXPLORER
# ──────────────────────────────────────────────────────────────────────────
elif page == "🗂️ Dataset Explorer":
    st.markdown("## Dataset Explorer")
    st.caption("Browse, search and filter the student dataset")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Rows", f"{len(df):,}", "students")
    with c2:
        metric_card("Columns", f"{df.shape[1]}", "features")
    with c3:
        metric_card("Null Values", f"{df.isnull().sum().sum()}", "clean dataset")
    with c4:
        metric_card("Duplicates", f"{df.duplicated().sum()}", "unique rows")

    st.write("")
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        search = st.text_input("Search", placeholder="Search by gender, ethnicity, education…")
    with f2:
        gender_filter = st.selectbox("Gender", ["All", "female", "male"])
    with f3:
        lunch_filter = st.selectbox("Lunch type", ["All", "standard", "free/reduced"])

    filtered = df.copy()
    if search:
        mask = filtered.apply(lambda row: row.astype(str).str.lower().str.contains(search.lower()).any(), axis=1)
        filtered = filtered[mask]
    if gender_filter != "All":
        filtered = filtered[filtered["gender"] == gender_filter]
    if lunch_filter != "All":
        filtered = filtered[filtered["lunch"] == lunch_filter]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} rows" + (" (filtered)" if len(filtered) < len(df) else ""))
    st.dataframe(filtered, hide_index=True, use_container_width=True, height=460)

# ──────────────────────────────────────────────────────────────────────────
# PAGE: EDA / CHARTS
# ──────────────────────────────────────────────────────────────────────────
elif page == "📈 EDA / Charts":
    st.markdown("## Exploratory Data Analysis")
    st.caption("Visual breakdown of how each factor affects student performance")

    section_label("By parental level of education")
    edu_avg = df.groupby("parental level of education")["math score"].mean().sort_values(ascending=True)
    fig = px.bar(
        x=edu_avg.values, y=[e.title() for e in edu_avg.index], orientation="h",
        color_discrete_sequence=[ORANGE],
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False)
    fig.update_xaxes(title="Average Math Score", gridcolor=BORDER)
    fig.update_yaxes(title=None, gridcolor=BORDER)
    st.plotly_chart(fig, use_container_width=True, key="edu_bars")

    e1, e2 = st.columns(2)
    with e1:
        st.caption("Average Score by Ethnicity Group")
        eth_avg = df.groupby("race/ethnicity")["math score"].mean().sort_index()
        fig = px.bar(
            x=[e.replace("group ", "Group ") for e in eth_avg.index], y=eth_avg.values,
            color_discrete_sequence=[ORANGE],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=240, showlegend=False)
        fig.update_xaxes(title=None, gridcolor=BORDER)
        fig.update_yaxes(title=None, gridcolor=BORDER)
        st.plotly_chart(fig, use_container_width=True, key="eth_chart")
    with e2:
        st.caption("Test Prep Course Impact")
        prep_avg = df.groupby("test preparation course")["math score"].mean().reindex(["none", "completed"])
        fig = px.bar(
            x=["None", "Completed"], y=prep_avg.values,
            color_discrete_sequence=[ORANGE],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=240, showlegend=False)
        fig.update_xaxes(title=None, gridcolor=BORDER)
        fig.update_yaxes(title=None, gridcolor=BORDER, range=[55, max(prep_avg.values) + 5])
        st.plotly_chart(fig, use_container_width=True, key="prep_chart")

    section_label("Score correlation heatmap")
    corr = df[["math score", "reading score", "writing score"]].corr().round(3)
    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale=["#1e293b", ORANGE],
        labels=dict(color="Correlation"),
        x=["Math", "Reading", "Writing"], y=["Math", "Reading", "Writing"],
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=320, width=420)
    st.plotly_chart(fig, key="heatmap")

    section_label("Boxplot — score spread & outliers")
    b1, b2, b3 = st.columns(3)
    for col, score_col, color, label in zip(
        [b1, b2, b3],
        ["math score", "reading score", "writing score"],
        [ORANGE, BLUE, GREEN],
        ["Math", "Reading", "Writing"],
    ):
        with col:
            st.caption(f"{label} Score Spread")
            fig = go.Figure(go.Box(y=df[score_col], marker_color=color, name=label))
            fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False)
            fig.update_yaxes(range=[0, 100], gridcolor=BORDER)
            st.plotly_chart(fig, use_container_width=True, key=f"box_{score_col}")

# ──────────────────────────────────────────────────────────────────────────
# PAGE: MODEL TRAINING
# ──────────────────────────────────────────────────────────────────────────
elif page == "🤖 Model Training":
    st.markdown("## Model Training & Comparison")
    st.caption("8 regression models trained live on 80% of the data, evaluated on the 20% holdout")

    best_row = results_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Best R² Score", f"{best_row['Test R²']:.3f}", best_row["Model"])
    with c2:
        metric_card("Best RMSE", f"{best_row['RMSE']:.2f}", best_row["Model"])
    with c3:
        metric_card("Best MAE", f"{best_row['MAE']:.2f}", best_row["Model"])
    with c4:
        metric_card("Models Tested", f"{len(results_df)}", "sklearn models")

    section_label("All model results (sorted by Test R² score)")
    display_df = results_df.copy()
    display_df.insert(0, "Rank", range(1, len(display_df) + 1))

    def perf_tag(r2):
        if r2 >= 0.86:
            return "🟢 Excellent"
        elif r2 >= 0.82:
            return "🔵 Good"
        elif r2 >= 0.78:
            return "🟠 Fair"
        return "🟡 Weak"

    display_df["Performance"] = display_df["Test R²"].apply(perf_tag)
    display_df["Train R²"] = display_df["Train R²"].round(4)
    display_df["Test R²"] = display_df["Test R²"].round(4)
    display_df["RMSE"] = display_df["RMSE"].round(2)
    display_df["MAE"] = display_df["MAE"].round(2)
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    section_label("Test R² score comparison")
    fig = px.bar(
        results_df.sort_values("Test R²"), x="Test R²", y="Model", orientation="h",
        color_discrete_sequence=[ORANGE], text="Test R²",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
    fig.update_xaxes(title=None, gridcolor=BORDER)
    fig.update_yaxes(title=None, gridcolor=BORDER)
    st.plotly_chart(fig, use_container_width=True, key="model_bars")

    section_label("Test R² vs RMSE scatter")
    scatter_df = results_df.copy()
    scatter_df["is_best"] = scatter_df["Model"] == BEST_MODEL_NAME
    fig = px.scatter(
        scatter_df, x="Test R²", y="RMSE", text="Model",
        color="is_best", color_discrete_map={True: ORANGE, False: BLUE},
        size=[14 if b else 9 for b in scatter_df["is_best"]],
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
    fig.update_xaxes(title="Test R² Score", gridcolor=BORDER)
    fig.update_yaxes(title="RMSE", gridcolor=BORDER)
    st.plotly_chart(fig, use_container_width=True, key="scatter_chart")

# ──────────────────────────────────────────────────────────────────────────
# PAGE: PREDICT SCORE
# ──────────────────────────────────────────────────────────────────────────
elif page == "⚡ Predict Score":
    st.markdown("## Predict Math Score")
    st.caption("Enter student details to get an ML-powered predicted math score")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Student Information")
        fc1, fc2 = st.columns(2)
        with fc1:
            p_gender = st.selectbox("Gender", ["female", "male"], format_func=str.capitalize)
        with fc2:
            p_eth = st.selectbox(
                "Race / Ethnicity", ["group A", "group B", "group C", "group D", "group E"],
                index=2, format_func=lambda x: x.replace("group ", "Group "),
            )
        p_edu = st.selectbox(
            "Parental Level of Education",
            ["some high school", "high school", "some college", "associate's degree",
             "bachelor's degree", "master's degree"],
            index=2, format_func=str.title,
        )
        fc3, fc4 = st.columns(2)
        with fc3:
            p_lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"], format_func=str.title)
        with fc4:
            p_prep = st.selectbox("Test Preparation Course", ["none", "completed"], format_func=str.title)

        p_read = st.slider("Reading Score", 0, 100, 70)
        p_write = st.slider("Writing Score", 0, 100, 70)

        model_options = results_df["Model"].tolist()
        model_labels = [
            f"{m} (Best — R² {results_df.loc[results_df['Model']==m,'Test R²'].values[0]:.3f})"
            if m == BEST_MODEL_NAME
            else f"{m} (R² {results_df.loc[results_df['Model']==m,'Test R²'].values[0]:.3f})"
            for m in model_options
        ]
        p_model_label = st.selectbox("ML Model", model_labels)
        p_model = model_options[model_labels.index(p_model_label)]

        predict_clicked = st.button("⚡ Predict Math Score", use_container_width=True, type="primary")

    with right:
        pipe = pipelines[p_model]
        input_row = pd.DataFrame([{
            "gender": p_gender,
            "race/ethnicity": p_eth,
            "parental level of education": p_edu,
            "lunch": p_lunch,
            "test preparation course": p_prep,
            "reading score": p_read,
            "writing score": p_write,
        }])
        pred = float(np.clip(pipe.predict(input_row)[0], 0, 100))

        if pred >= 90:
            grade, color = "A — Excellent", GREEN
        elif pred >= 80:
            grade, color = "B — Good", BLUE
        elif pred >= 70:
            grade, color = "C — Average", ORANGE
        elif pred >= 60:
            grade, color = "D — Below Average", YELLOW
        else:
            grade, color = "F — Needs Support", RED

        st.markdown(
            f"""<div class="result-card">
            <div style="font-size:12px;color:#475569;margin-bottom:6px;">{p_model}</div>
            <div class="result-score" style="color:{color};">{pred:.1f}</div>
            <div class="result-grade" style="color:{color};">{grade}</div>
            <div style="margin-top:16px;background:#162032;border-radius:8px;height:12px;overflow:hidden;">
                <div style="height:100%;border-radius:8px;background:{color};width:{pred}%;"></div>
            </div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("##### Feature insights")
        baseline_row = pd.DataFrame([{
            "gender": "female", "race/ethnicity": "group C",
            "parental level of education": "some college", "lunch": "standard",
            "test preparation course": "none", "reading score": p_read, "writing score": p_write,
        }])
        baseline_pred = float(pipe.predict(baseline_row)[0])
        st.caption(
            f"Compared to a baseline student (female, group C, some college, standard lunch, "
            f"no test prep) with the same reading/writing scores, this profile predicts "
            f"**{pred - baseline_pred:+.1f}** points on math score."
        )

    if predict_clicked:
        st.toast(f"Predicted score: {pred:.1f} using {p_model}", icon="⚡")
