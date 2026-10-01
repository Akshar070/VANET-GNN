from sklearn.cluster import KMeans, SpectralClustering

def run_kmeans(X, n_clusters, random_state=42):
    return KMeans(
        n_clusters=n_clusters,
        n_init=20,
        random_state=random_state
    ).fit_predict(X)

def run_spectral(affinity, n_clusters, random_state=42):
    return SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=random_state
    ).fit_predict(affinity)
