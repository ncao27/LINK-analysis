import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d


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



def thres_det(neural_data, threshold_multiplier = -4.5):
    '''
    Performs negative threshold detection

    neural_data: should be in the format time x channels
    threshold_multiplier: some float we multiply the rms with
    '''

    # take the rms and multiply by threshold_multiplier
    rms = np.sqrt(np.mean(neural_data ** 2, axis=0))
    thresholds = threshold_multiplier * rms

    # detect if we have a negative threshold
    below_threshold = neural_data < thresholds[None, :]

    # find the crossing_mask, regions where below threshold
    crossing_mask = below_threshold[1:] & ~below_threshold[:-1]

    # find specific time and channels based on crossing mask
    time_idx, channel_idx = np.where(crossing_mask)

    # find the specific crossing values
    crossings = np.column_stack([time_idx + 1, channel_idx])

    return crossings, thresholds

def smoothing(neural_data, sigma = 2):
    '''
    Performs smoothing

    neural_data: should be in the format time x channels
    sigma: variance of the gaussian function we smooth data with
    '''

    # smooth with Gaussian filter
    smoothed = gaussian_filter1d(
        neural_data,
        sigma=sigma,
        axis=0
    )

    return smoothed

def zscore_channels(neural_data):
    '''
    Normalize neural activity by channel

    neural_data: should be in the format time x channels
    '''

    # find the mean of the data by channel, across time
    mean = np.mean(neural_data, axis=0, keepdims=True)

    # find std of data by channel, across time
    std = np.std(neural_data, axis=0, keepdims=True)

    # normalize the data
    normalized = (neural_data - mean) / (std + 1e-8)

    return normalized




