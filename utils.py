import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans

def load_and_prep_data(filepath="nigeria_houses_data.csv"):
    df = pd.read_csv(filepath)
    
    # Filter for Lagos and Abuja as per the notebook logic
    df = df[df['state'].isin(['Lagos', 'Abuja'])]
    df = df.dropna()
    
    # Label Encoding for categorical columns
    le_town = LabelEncoder()
    le_state = LabelEncoder()
    df['town_encoded'] = le_town.fit_transform(df['town'])
    df['state_encoded'] = le_state.fit_transform(df['state'])
    
    # Price transformation
    df['price_log'] = np.log1p(df['price'])
    
    # Scaling numerical features
    features = ['bedrooms', 'bathrooms', 'parking_space', 'town_encoded', 'state_encoded', 'price_log']
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features])
    
    # Initialize and fit K-Means (K=5 based on the elbow method results)
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    df['cluster'] = kmeans.fit_predict(df_scaled)
    
    return df, kmeans, scaler, le_town, le_state
