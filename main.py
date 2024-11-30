# main.py
from models.kmeans_model import train_kmeans, plot_clusters_kmeans
from models.birch_model import train_birch, plot_clusters_birch
from evaluation.evaluate import evaluate_clustering
import pandas as pd

csv_path = 'data/clustering_dataset.csv'
df = pd.read_csv(csv_path)
X = df.values[:, :2]

# K-Means
kmeans, elapsed_kmeans, n_iter_kmeans, silhouette_kmeans, db_kmeans = train_kmeans(csv_path, n_clusters=3)
evaluate_clustering("K-Means", elapsed_kmeans, n_iter_kmeans, silhouette_kmeans, db_kmeans)
plot_clusters_kmeans(kmeans, X)

# BIRCH
birch, elapsed_birch, silhouette_birch, db_birch = train_birch(csv_path, n_clusters=3)
evaluate_clustering("BIRCH", elapsed_birch, None, silhouette_birch, db_birch)
plot_clusters_birch(birch, X)
