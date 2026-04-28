import numpy as np
import pandas as pd
import joblib
from scipy.spatial.distance import cdist

# Load artifacts once
kmeans   = joblib.load("models/kmeans_model.pkl")
scaler   = joblib.load("models/scaler.pkl")
le_state = joblib.load("models/le_state.pkl")
le_town  = joblib.load("models/le_town.pkl")
le_title = joblib.load("models/le_title.pkl")

df_new = pd.read_csv("data/nigeria_houses_clustered.csv")

feature_cols = [
    'bedrooms', 'bathrooms', 'toilets', 'parking_space',
    'log_price', 'state_enc', 'town_enc', 'title_enc'
]

def get_user_vector(bedrooms, bathrooms, toilets, parking_space,
                    budget, state, town, title):

    log_budget = np.log1p(budget)

    state_enc = le_state.transform([state])[0] if state in le_state.classes_ else 0
    town_enc  = le_town.transform([town])[0] if town in le_town.classes_ else 0
    title_enc = le_title.transform([title])[0] if title in le_title.classes_ else 0

    return np.array([[bedrooms, bathrooms, toilets, parking_space,
                      log_budget, state_enc, town_enc, title_enc]])


def recommend_by_cluster(bedrooms, bathrooms, toilets, parking_space,
                         budget_min, budget_max, state=None,
                         town=None, title=None, top_k=5):

    budget_mid = (budget_min + budget_max) / 2

    state = state or df_new['state'].mode()[0]
    town  = town or df_new['town'].mode()[0]
    title = title or df_new['title'].mode()[0]

    user_vec = get_user_vector(
        bedrooms, bathrooms, toilets, parking_space,
        budget_mid, state, town, title
    )

    user_scaled = scaler.transform(user_vec)
    user_cluster = kmeans.predict(user_scaled)[0]

    candidates = df_new[
        (df_new['cluster'] == user_cluster) &
        (df_new['price'] >= budget_min) &
        (df_new['price'] <= budget_max)
    ].copy()

    if candidates.empty:
        candidates = df_new[df_new['cluster'] == user_cluster].copy()

    # Feature similarity
    candidates_scaled = scaler.transform(candidates[feature_cols])
    distances = cdist(user_scaled, candidates_scaled, 'euclidean').flatten()

    candidates['feature_distance'] = distances
    candidates['price_diff'] = abs(candidates['price'] - budget_mid)

    top = candidates.sort_values(
        ['feature_distance', 'price_diff']
    ).head(top_k)

    display_cols = [
        'title', 'town', 'toilets', 'state',
        'bedrooms', 'bathrooms', 'parking_space', 'price'
    ]

    result = top[display_cols].copy()
    result['price'] = result['price'].apply(lambda x: f"₦{x:,.0f}")

    cluster_mean = df[df['cluster'] == user_cluster]['price'].mean()
    cluster_std = df[df['cluster'] == user_cluster]['price'].std()

    results['Price Flag'] = results['price'].apply(
        lambda p: '⚠️ Overpriced' if p > cluster_mean + 1.5 * cluster_std
        else '✅ Fair Price'
    )
                             
    return result
