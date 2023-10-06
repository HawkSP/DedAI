import logging
from concurrent.futures import ThreadPoolExecutor
from Data_Loading_and_Preprocessing.eeg_recording import EEGProcessor, record_eeg
from Data_Loading_and_Preprocessing.music_recording import MusicRecorder, record_music
from Data_Loading_and_Preprocessing.eeg_loading import EEGLoader
from Data_Loading_and_Preprocessing.eeg_preprocessing import EEGPreprocessor
from dedai_v2.cortex import Cortex
from dedai_v2.config import CLIENT_ID, CLIENT_SECRET, HEADSET_ID, CORTEX_TOKEN
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

def print_logo():
    print('''
     _          _  ___  _____ 
    | |        | |/ _ \|_   _|
  __| | ___  __| / /_\ \ | |  
 / _` |/ _ \/ _` |  _  | | |  
| (_| |  __/ (_| | | | |_| |_ 
 \__,_|\___|\__,_\_| |_/\___/
    ''')
    print("Copyright © 2023 Elliott Mitchell")
    print("All rights reserved. This software is proprietary and confidential.")
    print("Unauthorized copying of this file, via any medium is strictly prohibited.")

def main():
    print_logo()
    print("Initializing...")

    # Define user information
    user_info = {
        "clientId": CLIENT_ID,
        "clientSecret": CLIENT_SECRET,
        "headsetId": HEADSET_ID,
        "cortexToken": CORTEX_TOKEN
    }

    # Instantiate a Cortex object
    print("Starting Cortex...")
    cortex = Cortex(user_info)

    # Create a session
    print("Creating session...")
    session_result = cortex.create_session()
    assert 'result' in session_result, 'Failed to create session.'

    # Subscribe to the EEG data stream
    print("Subscribing to EEG data stream...")
    streams = ['eeg']
    subscribe_result = cortex.subscribe_data_stream(streams)
    assert 'result' in subscribe_result, 'Failed to subscribe to EEG data stream.'

    # Initialize an EEGProcessor
    print("Initializing EEG Processor...")
    eeg_processor = EEGProcessor(cortex.data_queue)

    # Create a MusicRecorder
    print("Initializing Music Recorder...")
    music_recorder = MusicRecorder()

    # Create a ThreadPoolExecutor
    print("Starting EEG and Music recording...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Start the EEG and music recording tasks
        eeg_future = executor.submit(record_eeg, cortex, eeg_processor, 60)
        music_future = executor.submit(record_music, music_recorder, 60)

        # Wait for both tasks to complete
        print("Recording...")
        for _ in tqdm(range(60), desc="Recording progress"):
            time.sleep(1)

        eeg_future.result()
        music_future.result()

    # Load the EEG data
    print("Loading EEG data...")
    eeg_loader = EEGLoader('eeg_data.csv')
    eeg_data, sample_rate = eeg_loader.load()

    # Preprocess the EEG data
    print("Preprocessing EEG data...")
    eeg_preprocessor = EEGPreprocessor(eeg_data, sample_rate)
    eeg_data = eeg_preprocessor.preprocess()
    print("Preprocessed EEG data:")
    print(eeg_data)

    # Close the session
    print("Closing session...")
    cortex.close_session()

    print("Done.")

if __name__ == "__main__":
    main()
