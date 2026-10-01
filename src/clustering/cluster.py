from sklearn.cluster import KMeans

def cluster_embeddings(embeddings, n_clusters, random_state=42):
    return KMeans(
        n_clusters=n_clusters,
        n_init=20,
        random_state=random_state
    ).fit_predict(embeddings)
