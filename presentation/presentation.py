import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Wczytaj dane świec (już przetworzone)
df = pd.read_parquet("artifacts/candles_features.parquet")

# Wybierz cechy numeryczne
features = df.select_dtypes(include=['float64', 'int64']).dropna()

# PCA
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(features)
df['PCA1'] = pca_result[:,0]
df['PCA2'] = pca_result[:,1]

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(features)

# Wykres
plt.figure(figsize=(8,6))
plt.scatter(df['PCA1'], df['PCA2'], c=df['cluster'], cmap='tab10', alpha=0.7)
plt.title("Clusters of candle features in PCA space")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.savefig("artifacts/candles_clusters.png")
plt.show()