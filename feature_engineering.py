from scipy.stats import kurtosis, skew
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
import numpy as np

def extract_features(data, sampling_rate=256):
    features = []
    for signal in data:
        mean = np.mean(signal)
        variance = np.var(signal)
        skewness = skew(signal)
        kurt = kurtosis(signal)
        freqs, psd = welch(signal, fs=sampling_rate)
        psd_mean = np.mean(psd)
        psd_variance = np.var(psd)
        feature_vector = [mean, variance, skewness, kurt, psd_mean, psd_variance]
        features.append(feature_vector)
    return np.array(features)

def one_hot_encode_labels(labels):
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    labels = labels.reshape(-1, 1)
    one_hot_labels = encoder.fit_transform(labels)
    return one_hot_labels, labels.ravel()

def apply_pca(features, n_components=None):
    if n_components is None or n_components > features.shape[1]:
        n_components = features.shape[1]
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    return reduced_features

def remove_constant_features(features, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    return selector.fit_transform(features)

def select_top_features(features, labels, k=5):
    selector = SelectKBest(score_func=f_classif, k="all")
    scores = selector.fit(features, labels).scores_

    valid_indices = ~np.isnan(scores)
    if np.sum(valid_indices) == 0:  # Fallback if all scores are NaN
        print("No valid scores computed. Returning original features.")
        return features

    filtered_features = features[:, valid_indices]
    filtered_scores = scores[valid_indices]

    k = min(k, filtered_features.shape[1])  # Dynamically adjust k
    top_k_indices = np.argsort(filtered_scores)[-k:]
    top_features = filtered_features[:, top_k_indices]

    print(f"Valid feature scores: {filtered_scores}")
    print(f"Selected top {k} features.")

    return top_features
