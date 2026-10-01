import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def validate_features(X):
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Invalid features: NaN or inf detected.")

def scale_features(X, method='standard'):
    validate_features(X)
    if method == 'standard':
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    return X

def run_kmeans(X, k, random_state=42, n_init=20, max_iter=300):
    """
    Run KMeans clustering on feature matrix X.
    """
    if len(X) < k:
        k = len(X)
        
    if k <= 1 or len(X) <= 1:
        return np.zeros(len(X), dtype=int), 0.0
        
    # All features are identical?
    if np.all(X == X[0]):
        return np.zeros(len(X), dtype=int), 0.0
        
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter)
    labels = kmeans.fit_predict(X)
    return labels, kmeans.inertia_
