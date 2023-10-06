# constants.py

import numpy as np


DELTA_BAND = (0, 4)  # Delta waves have a frequency range up to 4 Hz
THETA_BAND = (4, 7)  # Theta waves have a frequency range from 4 Hz to 7 Hz
ALPHA_BAND = (7, 13)  # Alpha waves have a frequency range from 7 Hz to 13 Hz
BETA_BAND = (14, 30)  # Beta waves have a frequency range from 14 Hz to 30 Hz
GAMMA_BAND = (30, np.inf)  # Gamma waves have a frequency range from 30 Hz and above


# Define EEG channels
EEG_CHANNELS = ['AF3', 'AF4', 'F3', 'F4', 'FC5', 'FC6', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2']

# Define EEG sampling rate
SAMPLE_RATE = 128  # Hz (user configured)

# Define file paths
EEG_DATA_PATH = 'eeg_data.csv'
MUSIC_DATA_PATH = 'music_data.csv'

# Define artifact removal constants
ICA_COMPONENTS = 15
ICA_RANDOM_STATE = 97

# Define powerline frequency (depends on your location)
POWERLINE_FREQ = 50  # Hz

# Paths to important resources
MODEL_PATH = 'models/'
LOG_PATH = 'logs/'
