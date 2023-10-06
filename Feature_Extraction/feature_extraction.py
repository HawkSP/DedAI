import numpy as np
import pandas as pd
from scipy.signal import welch
from dedai_v2.Utils_and_Constants.constants import DELTA_BAND, THETA_BAND, ALPHA_BAND, BETA_BAND, GAMMA_BAND, SAMPLE_RATE

class EEGFeatureExtractor:
    def __init__(self, eeg_data, window_length=1):
        self.eeg_data = eeg_data
        self.features = pd.DataFrame()
        self.window_length = window_length  # Length of time window in seconds

    def power_spectral_density(self, channel_data):
        """Compute the power spectral density for a given EEG channel."""
        freqs, psd = welch(channel_data, fs=SAMPLE_RATE)
        return freqs, psd

    def band_power(self, psd, freqs, band):
        """Compute the band power for a given frequency band."""
        band_freqs = (freqs >= band[0]) & (freqs <= band[1])
        return np.sum(psd[band_freqs])

    def compute_features(self, data):
        """Compute features for a given window of EEG data."""
        features = {}
        for channel in data.columns:
            freqs, psd = self.power_spectral_density(data[channel])
            features[channel + '_delta_power'] = self.band_power(psd, freqs, DELTA_BAND)
            features[channel + '_theta_power'] = self.band_power(psd, freqs, THETA_BAND)
            features[channel + '_alpha_power'] = self.band_power(psd, freqs, ALPHA_BAND)
            features[channel + '_beta_power'] = self.band_power(psd, freqs, BETA_BAND)
            features[channel + '_gamma_power'] = self.band_power(psd, freqs, GAMMA_BAND)
        return features

    def extract_features(self):
        """Extract features from the EEG data."""
        num_samples = len(self.eeg_data)
        window_length_samples = self.window_length * SAMPLE_RATE

        # Slide the window across the EEG data
        for i in range(0, num_samples, window_length_samples):
            window_data = self.eeg_data.iloc[i:i+window_length_samples]
            window_features = self.compute_features(window_data)
            self.features = self.features.append(window_features, ignore_index=True)

        # Compute the change in band power between successive time windows
        self.features = self.features.diff().dropna()

        return self.features

    def identify_large_shifts(self, threshold=1.0):
        """Identify large shifts in band power."""
        # Extract features
        self.extract_features()

        # Compute the absolute difference in band power between successive time windows
        diff = self.features.diff().abs()
        self.features = diff.dropna()

        # Identify large shifts (where the difference exceeds the threshold)
        large_shifts = diff > threshold

        # Add large shift indicators to the features DataFrame
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            self.features['large_shift_' + band] = large_shifts[band + '_power']

        return self.features

    def extract_and_save_features(self, eeg_file_path, music_file_path):
        """Extract features from the EEG and music data and save to CSV files."""
        # Identify large shifts
        self.identify_large_shifts()

        # Save features to CSV files
        self.save_features(eeg_file_path)

        # To be implemented: Extract features from music data and save to CSV file

    def save_features(self, file_path):
        """Save the extracted features to a CSV file."""
        self.features.to_csv(file_path, index=False)
