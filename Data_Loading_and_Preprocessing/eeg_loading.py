import pandas as pd

class EEGLoader:
    def __init__(self, file_path):
        try:
            self.data = pd.read_csv(file_path)
            self.sample_rate = 128  # Hz, adapt this to your actual sampling rate
        except FileNotFoundError as e:
            print(f"File not found: {file_path}")
            raise e

    def load(self):
        return self.data, self.sample_rate
