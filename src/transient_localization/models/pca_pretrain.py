import numpy as np
from sklearn.decomposition import PCA


def apply_pca(x_train, n_components=10):
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    return pca, x_train_pca
