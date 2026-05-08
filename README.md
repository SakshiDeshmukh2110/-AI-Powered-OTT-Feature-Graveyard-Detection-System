# -AI-Powered-OTT-Feature-Graveyard-Detection-System
📌 Project Overview

This project is an AI-powered OTT analytics platform designed to identify unhealthy, risky, and declining OTT platform features using:

* Product Analytics
* Funnel Analytics
* NLP
* Machine Learning
* Interactive Streamlit Dashboard
The system predicts whether a feature is:

* ✅ Healthy
  
* ⚠️ At Risk
  
* ❌ Dead

Based on:
* engagement
* retention
* churn
* rage clicks
* funnel drop-offs
* customer complaints
* sentiment analysis

🎯 Business Problem

OTT platforms launch multiple product features such as:

* Watch Party
* Downloads
* Watchlist
* AI Recommendations
* Mini Preview Clips

However, companies often struggle to determine:

* which features users love
* which features users abandon
* which features create frustration
* which features are slowly dying
This project helps OTT companies make data-driven product decisions by detecting feature health using analytics and AI.

🚀 Key Features
✅ Product Analytics

Analyzes:

* Engagement
* Retention
* Churn
* Watch time
* Feature reuse

✅ Funnel Analytics

Detects:

* Rage clicks
* Retries
* Drop-offs
* Errors
* Friction points

✅ NLP & Complaint Intelligence

Uses:

* Sentiment analysis
* TF-IDF
* Complaint clustering
* Complaint classification
* Ratings analysis

✅ Machine Learning Prediction Engine

Predicts:

* Healthy Features
* At Risk Features
* Dead Features

using:

* Random Forest Classifier

✅ Interactive Streamlit Dashboard

Includes:

* KPI cards
* Funnel charts
* NLP visualizations
* Feature health simulation
* Prediction engine
* Interactive graphs

  🛠️ Technologies Used
  Category	Technologies
Programming	Python
Analytics	Pandas, NumPy
Visualization	Plotly, Matplotlib, Seaborn
NLP	TextBlob, TF-IDF
ML	Scikit-learn, Random Forest
Dashboard	Streamlit

📂 Project Structure
```OTT_Feature_Graveyard_Detector/
│
├── graveyard.py
├── requirements.txt
├── README.md
│
├── feature_usage_dataset.csv
├── funnel_events_dataset.csv
└── reviews_complaints_dataset.csv
```

📊 Datasets Used

1️⃣ Feature Usage Dataset

Contains:

* engagement_score
* retention_30d
* churn_flag
* watch_time_minutes
* reused_feature
  
2️⃣ Funnel Events Dataset

Contains:

* rage_clicks
* retries
* dropoff_flag
* error_flag
* stage_completed
  
3️⃣ Reviews & Complaints Dataset

Contains:

*sentiment
* ratings
* support tickets
* complaint categories
* review text

  
🤖 Machine Learning Workflow
Step 1
Aggregate feature metrics

Step 2
Merge product, funnel, and NLP datasets

Step 3
Create feature health labels

Step 4
Train Random Forest model

Step 5
Predict feature health

📈 Feature Health Categories

Status	Meaning

✅ Healthy	Strong engagement and retention

⚠️ At Risk	Moderate churn and weak UX signals

❌ Dead	High churn, complaints, and rage clicks

🧠 Example Predictions

Feature	Prediction
Downloads	Healthy
Watchlist	At Risk
Watch Party	Dead

▶️ Run Locally

Install Requirements
pip install -r requirements.txt
Run Streamlit App
streamlit run graveyard.py

🌐 Streamlit Deployment

Deploy easily on:

https://share.streamlit.io

📌 Future Improvements

XGBoost Integration
Real-time OTT logs
Live sentiment ingestion
Advanced recommendation engine
Feature death risk scoring
Cloud deployment

🏆 Key Learnings

Product Analytics
Funnel Analytics
NLP
Machine Learning
Streamlit Deployment
Business Storytelling
Interactive Dashboard Development

👩‍💻 Author

Sakshi Deshmukh
