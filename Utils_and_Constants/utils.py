# utils.py
import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt, iirnotch

def normalize(data):
    """Normalize data to zero mean and unit standard deviation."""
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a bandpass filter to the data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def notch_filter(data, freq, fs, Q=30):
    """Apply a notch filter at a specified frequency."""
    nyq = 0.5 * fs
    freq = freq / nyq
    b, a = iirnotch(freq, Q)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def frequency_transform(data):
    """Apply FFT to data and keep only the positive frequencies (up to Nyquist frequency)."""
    transformed_data = np.abs(fft(data, axis=0)) / len(data)
    return transformed_data[:len(transformed_data)//2]

def band_power(data, fs, band):
    """Calculate power in a specific frequency band."""
    f_vals = fftfreq(len(data), 1.0/fs)
    f_vals = f_vals[:len(f_vals)//2]  # Only keep positive frequencies
    idx_band = np.logical_and(f_vals >= band[0], f_vals <= band[1])
    return np.sum(frequency_transform(data)[idx_band])

def dominant_frequency(data, fs):
    """Find the frequency with the highest power in the spectrum."""
    f_vals = fftfreq(len(data), 1.0/fs)
    f_vals = f_vals[:len(f_vals)//2]  # Only keep positive frequencies
    return f_vals[np.argmax(frequency_transform(data))]
