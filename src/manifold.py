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


