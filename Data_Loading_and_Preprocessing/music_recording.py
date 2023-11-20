
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time
import logging
from threading import Thread, Event
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure the duration of the recording and the sample rate
DURATION = 60  # seconds
SAMPLE_RATE = 44100  # Hz
CONFIG_FILE = "data/settings/config.json"


class MusicRecorder:
    def __init__(self, stop_event):
        # Load configuration
        self.config = self.load_config()
        self.device_index = self.config.get("device_index", None)

        # Create a buffer to store the audio data
        self.buffer = np.array([])
        self.stop_event = stop_event

        # Start recording audio
        self.stream = sd.InputStream(callback=self.audio_callback, channels=1, samplerate=SAMPLE_RATE,
                                     device=self.device_index)
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        # Append the incoming audio data to the buffer
        self.buffer = np.append(self.buffer, indata)

        # Check if we need to stop recording
        if self.stop_event.is_set():
            raise sd.CallbackStop()

    def save(self, filename):
        # Save the buffer to a WAV file
        wav.write(filename, SAMPLE_RATE, self.buffer)

    @staticmethod
    def load_config():
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as file:
                return json.load(file)
        return {"device_index": None}

    @staticmethod
    def save_config(config):
        with open(CONFIG_FILE, "w") as file:
            json.dump(config, file)


def record_music(recorder, duration):
    # Wait for the desired duration
    start_time = time.time()
    while time.time() - start_time < duration:
        time.sleep(0.1)  # Sleep for a short while to reduce CPU usage
        if recorder.stop_event.is_set():
            logging.info("Stopping music recording...")
            break

    # Save the recording
    recorder.save('music.wav')

def main():
    # Create a stop event
    stop_event = Event()

    # Create a MusicRecorder
    recorder = MusicRecorder(stop_event)

    # Create a thread for the music recording
    music_thread = Thread(target=record_music, args=(recorder, DURATION))

    # Start the thread
    music_thread.start()

    try:
        # Let the recording continue while we wait for user input
        input("Press Enter to stop recording...")
    except KeyboardInterrupt:
        pass
    finally:
        # Set the stop event to stop the recording
        stop_event.set()
        # Wait for the recording thread to finish
        music_thread.join()

if __name__ == "__main__":
    main()
