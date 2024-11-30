# models/kmeans_model.py
import time
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt


def train_kmeans(csv_path, n_clusters=3):
    df = pd.read_csv(csv_path)
    X = df.values

    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X)
    elapsed_time = time.time() - start_time

    silhouette = silhouette_score(X, kmeans.labels_)
    davies_bouldin = davies_bouldin_score(X, kmeans.labels_)

    return kmeans, elapsed_time, kmeans.n_iter_, silhouette, davies_bouldin


def plot_clusters_kmeans(kmeans, X):
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=50)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')
    plt.title("K-Means Clustering")
    plt.show()
