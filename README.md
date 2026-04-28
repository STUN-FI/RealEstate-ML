# EstateML
### Intelligent Property Recommendation System for Nigeria

> Live Demo: https://smartpropertyrecommender.streamlit.app/

---

## The Problem
Nigerian property platforms show thousands of listings with little personalisation, making it harder for users to quickly find suitable homes.

## The Solution
EstateML uses machine learning to recommend properties based on user preferences such as location, budget, bedrooms, bathrooms, and parking space.

---

## Screenshots
Add screenshot of your running Streamlit app here.

---

## Architecture
Data → Preprocessing → KMeans Clustering → Distance Scoring → Ranked Recommendations → Streamlit UI

---

## Model Evaluation
- Silhouette Score: To be added after running `evaluate.py`
- Elbow method used to validate cluster selection
- Elbow plot: To be added after running `evaluate.py`

---

## Dataset
Nigeria housing dataset filtered for Lagos and Abuja properties.

---

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
