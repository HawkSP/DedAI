import time
import logging
import pandas as pd
import numpy as np
from queue import Empty
from scipy.signal import butter, filtfilt
from threading import Thread
from dedai_v2.cortex import Cortex
from dedai_v2 import config

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import constants
from dedai_v2.Utils_and_Constants.constants import DELTA_BAND, THETA_BAND, ALPHA_BAND, BETA_BAND, GAMMA_BAND, SAMPLE_RATE

# Define EEG frequency bands
ALPHA_BAND = ALPHA_BAND

class EEGProcessor:
    def __init__(self, data_queue):
        self.data_queue = data_queue
        self.eeg_data = pd.DataFrame()

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    def process(self):
        while True:
            try:
                data = self.data_queue.get(timeout=1)
                # Apply a bandpass filter to keep only the alpha waves
                data = self.bandpass_filter(data, ALPHA_BAND[0], ALPHA_BAND[1], fs=128)
                self.eeg_data = self.eeg_data.append(data, ignore_index=True)
            except Empty:
                break

    def save(self):
        self.eeg_data.to_csv('eeg_data.csv', index=False)

def record_eeg(cortex, processor, duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        processor.process()

    # Save the EEG data to a CSV file
    processor.save()

    # Unsubscribe from the data stream
    cortex.unsubscribe_data_stream(streams)

    # Close the session
    cortex.close_session()

def main():
    # Define user information
    user_info = {
        "clientId": config.CLIENT_ID,
        "clientSecret": config.CLIENT_SECRET,
        "headsetId": config.HEADSET_ID,
        "cortexToken": config.CORTEX_TOKEN
    }

    # Instantiate a Cortex object
    try:
        cortex = Cortex(user_info)

        # Create a session
        session_result = cortex.create_session()
        assert 'result' in session_result, 'Failed to create session.'

        # Subscribe to the EEG data stream
        streams = ['eeg']
        subscribe_result = cortex.subscribe_data_stream(streams)
        assert 'result' in subscribe_result, 'Failed to subscribe to EEG data stream.'

        # Initialize an EEGProcessor
        processor = EEGProcessor(cortex.data_queue)

        # Create a thread for the EEG recording
        eeg_thread = Thread(target=record_eeg, args=(cortex, processor, 60))

        # Start the thread
        eeg_thread.start()

    except Exception as e:
        logging.error(str(e))
    finally:
        if cortex:
            cortex.close_session()

if __name__ == "__main__":
    main()
