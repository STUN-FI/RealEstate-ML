from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Load your data
df = pd.read_csv('data/nigeria_houses_data.csv')

# Select features
X = df[['price', 'bedrooms', 'bathrooms', 'parking_space']]

# Train KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Silhouette Score
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.3f}")

# Elbow Method
inertias = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.savefig('elbow_plot.png')

print("Elbow plot saved.")
