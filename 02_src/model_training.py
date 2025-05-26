# Importar librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.cargar_datos import load_data

# Cargar los datos
train_data = load_data('data/00 - wine-clustering.csv')

# Entrenar modelo con hiperparámetros definidos (k=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_clusters = kmeans.fit_predict(train_data)

# Realizar PCA para visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(train_data)

# Graficar la solución
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=kmeans_clusters, palette="Set1", s=60)
plt.title("KMeans Clustering (3 clusters)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente principal 2")
plt.legend(title="Cluster")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

## Cálculo de Métricas de Evaluación de Clustering

# Silhouette Score
silhouette_avg = silhouette_score(train_data, kmeans_clusters)
print(f"Silhouette Score (k=3): {silhouette_avg:.3f}")

# Davies-Bouldin Score
davies_bouldin_avg = davies_bouldin_score(train_data, kmeans_clusters)
print(f"Davies-Bouldin Score (k=3): {davies_bouldin_avg:.3f}")

# Calinski-Harabasz Score
calinski_harabasz_avg = calinski_harabasz_score(train_data, kmeans_clusters)
print(f"Calinski-Harabasz Score (k=3): {calinski_harabasz_avg:.3f}")