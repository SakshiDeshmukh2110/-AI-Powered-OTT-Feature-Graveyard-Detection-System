
# =========================================================
# AI POWERED OTT FEATURE GRAVEYARD DETECTOR
# STREAMLIT DASHBOARD
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="OTT Feature Graveyard Detector",
    page_icon="🎬",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a,
            #111827,
            #1e293b,
            #312e81
        );
        color: white;
    }

    h1, h2, h3, h4, h5 {
        color: white !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111827,
            #1e1b4b
        );
    }

    .metric-card {
        padding: 20px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #2563eb,
            #7c3aed
        );
        color: white;
        text-align: center;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    }

    .prediction-box {
        padding: 25px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #059669,
            #14b8a6
        );
        color: white;
        text-align: center;
        font-size: 30px;
        font-weight: bold;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TITLE
# =========================================================

st.title("🎬 AI Powered OTT Feature Graveyard Detector")

st.markdown("""
### Detect unhealthy OTT features using:
- Product Analytics
- Funnel Analytics
- NLP
- Machine Learning
""")

# =========================================================
# LOAD DATASETS
# =========================================================

feature_df = pd.read_csv(
    "feature_usage_dataset.csv"
)

funnel_df = pd.read_csv(
    "funnel_events_dataset.csv"
)

nlp_df = pd.read_csv(
    "reviews_complaints_dataset.csv"
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Dashboard Controls")

selected_feature = st.sidebar.selectbox(
    "Select Feature",
    feature_df["feature_name"].unique()
)

selected_country = st.sidebar.selectbox(
    "Select Country",
    feature_df["country"].unique()
)

selected_device = st.sidebar.selectbox(
    "Select Device",
    feature_df["device_type"].unique()
)

# =========================================================
# FILTER DATA
# =========================================================

filtered_feature_df = feature_df[
    (feature_df["feature_name"] == selected_feature)
    &
    (feature_df["country"] == selected_country)
    &
    (feature_df["device_type"] == selected_device)
]

# =========================================================
# KPI METRICS
# =========================================================

avg_engagement = round(
    filtered_feature_df["engagement_score"].mean(),
    2
)

avg_retention = round(
    filtered_feature_df["retention_30d"].mean(),
    2
)

avg_churn = round(
    filtered_feature_df["churn_flag"].mean(),
    2
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        f"""
        <div class='metric-card'>
        <h3>🔥 Engagement</h3>
        <h1>{avg_engagement}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class='metric-card'>
        <h3>📈 Retention</h3>
        <h1>{avg_retention}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        f"""
        <div class='metric-card'>
        <h3>⚠️ Churn</h3>
        <h1>{avg_churn}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# =========================================================
# FEATURE HEALTH ANALYSIS
# =========================================================

st.header("📊 Feature Health Analysis")

feature_health = (

    feature_df.groupby("feature_name")
    .agg({

        "engagement_score": "mean",

        "retention_30d": "mean",

        "churn_flag": "mean"

    })

    .reset_index()

)

fig1 = px.bar(
    feature_health,
    x="feature_name",
    y="engagement_score",
    color="retention_30d",
    template="plotly_dark",
    title="Feature Engagement vs Retention"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# =========================================================
# FUNNEL ANALYSIS
# =========================================================

st.header("🚀 Funnel Analytics")

funnel_conversion = (

    funnel_df.groupby("funnel_stage")[
        "stage_completed"
    ]
    .sum()
    .reset_index()

)

fig2 = go.Figure(

    go.Funnel(

        y=funnel_conversion["funnel_stage"],

        x=funnel_conversion["stage_completed"],

        textinfo="value+percent initial"

    )

)

fig2.update_layout(
    template="plotly_dark",
    title="OTT Funnel Conversion"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =========================================================
# RAGE CLICK ANALYSIS
# =========================================================

st.header("😡 Rage Click Analysis")

rage_analysis = (

    funnel_df.groupby("feature_name")[
        "rage_clicks"
    ]
    .mean()
    .reset_index()

)

fig3 = px.bar(
    rage_analysis,
    x="feature_name",
    y="rage_clicks",
    color="rage_clicks",
    template="plotly_dark",
    title="Average Rage Clicks"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# =========================================================
# NLP ANALYSIS
# =========================================================

st.header("🧠 NLP Complaint Intelligence")

sentiment_counts = (

    nlp_df["sentiment"]
    .value_counts()
    .reset_index()

)

sentiment_counts.columns = [
    "sentiment",
    "count"
]

fig4 = px.pie(
    sentiment_counts,
    names="sentiment",
    values="count",
    template="plotly_dark",
    title="Sentiment Distribution"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# =========================================================
# BUILD MASTER DATASET
# =========================================================

funnel_metrics = (

    funnel_df.groupby("feature_name")
    .agg({

        "rage_clicks": "mean",

        "retries": "mean",

        "dropoff_flag": "mean",

        "error_flag": "mean",

        "stage_completed": "mean"

    })

    .reset_index()

)

nlp_metrics = (

    nlp_df.groupby("feature_name")
    .agg({

        "rating": "mean",

        "support_ticket_flag": "mean"

    })

    .reset_index()

)

feature_metrics = (

    feature_df.groupby("feature_name")
    .agg({

        "engagement_score": "mean",

        "retention_30d": "mean",

        "churn_flag": "mean",

        "reused_feature": "mean",

        "watch_time_minutes": "mean"

    })

    .reset_index()

)

master_df = pd.merge(
    feature_metrics,
    funnel_metrics,
    on="feature_name"
)

master_df = pd.merge(
    master_df,
    nlp_metrics,
    on="feature_name"
)

# =========================================================
# TARGET LABELS
# =========================================================

health_map = {

    "Downloads": "Healthy",

    "AI Recommendations": "Healthy",

    "Watchlist": "At Risk",

    "Mini Preview Clips": "At Risk",

    "Watch Party": "Dead"
}

master_df["feature_health"] = (

    master_df["feature_name"]
    .map(health_map)

)

# =========================================================
# ENCODE TARGET
# =========================================================

encoder = LabelEncoder()

master_df["target"] = encoder.fit_transform(
    master_df["feature_health"]
)

# =========================================================
# FEATURES & TARGET
# =========================================================

X = master_df[[

    "engagement_score",

    "retention_30d",

    "churn_flag",

    "reused_feature",

    "watch_time_minutes",

    "rage_clicks",

    "retries",

    "dropoff_flag",

    "error_flag",

    "stage_completed",

    "rating",

    "support_ticket_flag"

]]

y = master_df["target"]

# =========================================================
# TRAIN MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# =========================================================
# PREDICTION ENGINE
# =========================================================

st.header("🤖 AI Graveyard Prediction Engine")

engagement_input = st.slider(
    "Engagement Score",
    0,
    100,
    50
)

retention_input = st.slider(
    "30D Retention",
    0.0,
    1.0,
    0.5
)

churn_input = st.slider(
    "Churn Rate",
    0.0,
    1.0,
    0.3
)

rage_input = st.slider(
    "Rage Clicks",
    0.0,
    10.0,
    3.0
)

rating_input = st.slider(
    "Feature Rating",
    1.0,
    5.0,
    3.5
)

prediction_data = np.array([[

    engagement_input,

    retention_input,

    churn_input,

    0.5,

    40,

    rage_input,

    2,

    0.4,

    0.2,

    0.7,

    rating_input,

    0.3

]])

prediction = model.predict(
    prediction_data
)

prediction_label = encoder.inverse_transform(
    prediction
)[0]

st.markdown(
    f"""
    <div class='prediction-box'>
    Predicted Feature Health:<br><br>
    {prediction_label}
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.header("📌 Feature Importance")

importance = pd.DataFrame({

    "feature": X.columns,

    "importance": model.feature_importances_

})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

fig5 = px.bar(
    importance,
    x="importance",
    y="feature",
    orientation="h",
    color="importance",
    template="plotly_dark",
    title="Feature Importance"
)

st.plotly_chart(
    fig5,
    use_container_width=True
)

# =========================================================
# EXECUTIVE INSIGHTS
# =========================================================

st.header("📌 Executive Insights")

st.success(
    """
    Downloads and AI Recommendations
    are the healthiest OTT features.

    Watch Party demonstrates:
    - high churn
    - low retention
    - high rage clicks
    - strong negative sentiment

    Funnel friction and customer complaints
    strongly influence feature death risk.
    """
)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    ### 🎯 Technologies Used

    - Product Analytics
    - Funnel Analytics
    - NLP
    - Machine Learning
    - Streamlit
    - Random Forest
    """
)
