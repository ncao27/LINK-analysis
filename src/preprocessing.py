import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA


def car(neural_data):
    '''
    Performs common average referencing.

    neural_data: should be in the format time x channels
    '''

    # take the avg across time, independently for channels
    # axis = 1 bc we remove shared noise across channels
    common_avg = np.mean(neural_data, axis = 1, keepdims = True)

    # subtract the common_avg from original data
    car_data = neural_data - common_avg

    return car_data



def bandpass(fs, low_fc, high_fc, neural_data, order = 4):
    '''
    Performs bandpass filtering given lower and upper frequencies

    fs: sampling frequency
    low_fc: lower cutoff frequency
    high_fc: higher cutoff frequency
    neural_data: should be in the format time x channels
    order: represents order of butterworth filter

    butterworth: we do bandpass filtering with a butterworth (by default 4th order) filter
    filtfilt: use filtfilt function to perform zero-phase digital filtering (ie 
        ensure that after butterworth there are no phase distortions) 
    '''

    # calculate nyquist frequency
    nq = fs / 2

    # normalize lower and higher frequencies with nyquist frequency
    low = low_fc / nq
    high = high_fc / nq

    # perform butterworth filtering and zero-phase correction
    b, a = butter(order, [low, high], btype = "bandpass")
    filtered_data = filtfilt(b, a, neural_data, axis = 0)

    return filtered_data



def whitening(neural_data, e = 10e-6):
    '''
    Performs whitening, time-wise, across channels. So basically at each time point
    we remove cross-correlations in between channels.

    neural_data: should be in the format time x channels
    e = some small number to ensure nonzero denominator
    '''

    # center data across time (note that CAR removes noise across channels)
    centered = neural_data = np.mean(neural_data, axis = 0, keepdims = True)

    # find the covariance matrix
    cov = np.cov(centered, rowvar = False)

    # find eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov)

    # get the actual whitening matrix
    whitening_matrix = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + e)) @ eigvecs.T

    # whiten the data
    whitened_data = centered @ whitening_matrix

    return whitened_data



def thres_det():
    '''
    Performs threshold detection
    '''

def spike_sorting():
    '''
    Performs spike sorting
    '''

def binning():
    '''
    Performs binning
    '''

def smoothening():
    '''
    Performs smoothening
    '''