# models/birch_model.py
import time
import pandas as pd
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt


def train_birch(csv_path, n_clusters=3):
    df = pd.read_csv(csv_path)
    X = df.values

    start_time = time.time()
    birch = Birch(n_clusters=n_clusters)
    birch.fit(X)
    elapsed_time = time.time() - start_time

    silhouette = silhouette_score(X, birch.labels_)
    davies_bouldin = davies_bouldin_score(X, birch.labels_)

    return birch, elapsed_time, silhouette, davies_bouldin


def plot_clusters_birch(birch, X):
    plt.scatter(X[:, 0], X[:, 1], c=birch.labels_, cmap='coolwarm', s=50)
    plt.title("BIRCH Clustering")
    plt.show()
