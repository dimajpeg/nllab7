# evaluation/evaluate.py
def evaluate_clustering(model_name, elapsed_time, n_iter, silhouette, davies_bouldin):
    print(f"\nМодель: {model_name}")
    print(f"Время работы: {elapsed_time:.2f} секунд")
    if n_iter:
        print(f"Число итераций: {n_iter}")
    print(f"Silhouette Score: {silhouette:.2f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.2f}")
