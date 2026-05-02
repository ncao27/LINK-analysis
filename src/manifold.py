import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import orthogonal_procrustes


def procrustes_alignment(X, Z_target, n_components = 10):
    '''
    Dimensionality reduction using PCA on the manifold. Also learns the rotation matrix
    that maps target manifold (done through PCA) onto source manifold

    X: spiking band power data
    After PCA we return something of dim 10 instead of dimension 96
    '''

    # declare PCA object 
    pca = PCA(n_components = n_components)

    # grab 10 principle components with pca
    Z = pca.fit_transform(X)

    # find minimum length
    min_len = min(len(Z), len(Z_target))

    # find the matrices
    A = Z_target[:min_len]
    B = Z[:min_len]

    # find the orthogonal transformation
    R, scale = orthogonal_procrustes(A, B)

    return Z_target @ R * scale, R, scale

def manifold_alignment(
    X_src,
    y_src,
    X_tgt,
    y_tgt,
    neural_key = "SpikingBandPower",
    behavior_key = "index_position",
    n_components = 10,
    smoothing_sigma = 2,
    alpha = 1.0
):
    '''
    Perform manifold alignment. This is done on data across sessions in hopes of learning
    noise, electrodes moving, and neuroplasticity so that we can have higher cross-
    session prediction accuracy.

    Since preprocessing is done in the jupyter notebook, we assume that
    X_src, y_src, X_tgt, y_tgt have already been normalized and preprocessed

    For now we only care about Spiking Band Power, and we use that to predict index position
    '''
    return



