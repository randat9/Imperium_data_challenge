import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Upewnij się, że katalog artifacts istnieje
os.makedirs("artifacts", exist_ok=True)

# 2. Wczytaj dane (spróbuj z odpowiednim kodowaniem i separatorem)
df = pd.read_csv("XAUUSD_5min_candles.csv", encoding="utf-16", sep=";")

# 3. Konwersja kolumn na liczby
numeric_cols = ['Open','High','Low','Close']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

features = df[numeric_cols].dropna()

# 4. PCA
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(features)
df['PCA1'] = pca_result[:,0]
df['PCA2'] = pca_result[:,1]

# 5. Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(features)

# 6. Wykres
plt.figure(figsize=(8,6))
plt.scatter(df['PCA1'], df['PCA2'], c=df['cluster'], cmap='tab10', alpha=0.7)
plt.title("Clusters of candles in PCA space")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.savefig("artifacts/candles_clusters.png")
plt.show()