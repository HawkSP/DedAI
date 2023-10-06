import numpy as np
from scipy.fftpack import fft
from scipy.signal import iirnotch
from mne.preprocessing import ICA, create_eog_epochs
from mne import create_info
from mne.io import RawArray
from sklearn.decomposition import FastICA

# Import constants
from dedai_v2.Utils_and_Constants.constants import DELTA_BAND, THETA_BAND, ALPHA_BAND, BETA_BAND, GAMMA_BAND, SAMPLE_RATE


# Define EEG frequency bands
DELTA_BAND = DELTA_BAND
THETA_BAND = THETA_BAND
ALPHA_BAND = ALPHA_BAND
BETA_BAND = BETA_BAND
GAMMA_BAND = GAMMA_BAND

class EEGPreprocessor:
    def __init__(self, data, sample_rate=SAMPLE_RATE):
        self.data = data
        self.sample_rate = sample_rate

    def remove_artifacts(self):
        # Convert DataFrame to MNE Raw object
        info = create_info(ch_names=list(self.data.columns), sfreq=self.sample_rate, ch_types='eeg')
        raw = RawArray(self.data.transpose().values, info)

        # Apply ICA
        ica = ICA(n_components=15, random_state=97)
        ica.fit(raw)

        # Find EOG artifacts
        eog_epochs = create_eog_epochs(raw, ch_name='Fp1')  # adapt 'Fp1' to the name of your EOG channel
        eog_inds, scores = ica.find_bads_eog(eog_epochs)

        # Remove EOG artifacts
        ica.exclude = eog_inds
        self.data = ica.apply(self.data)

        # Apply a bandpass filter to keep frequencies in the EEG range (0.5-45 Hz)
        raw.filter(0.5, 45)

        # Create an array from the raw object
        self.data = raw.get_data().T

    def normalize(self):
        # Normalize each channel
        self.data = (self.data - np.mean(self.data, axis=0)) / np.std(self.data, axis=0)

    def frequency_transform(self):
        # Apply FFT to each channel
        self.data = np.abs(fft(self.data, axis=0)) / len(self.data)
        # Keep only the positive frequencies (up to Nyquist frequency)
        self.data = self.data[:len(self.data)//2]

    def preprocess(self):
        self.remove_artifacts()
        self.normalize()
        self.frequency_transform()
        return self.data
