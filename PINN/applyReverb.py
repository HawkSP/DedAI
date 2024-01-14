import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import fftconvolve, butter, sosfilt
import random

# Read the CSV file containing genre, IR paths, and wet/dry ratios
def read_reverb_settings(csv_path):
    return pd.read_csv(csv_path)

# Select a random IR from the specified genre
def select_random_ir(reverb_settings, genre):
    genre_settings = reverb_settings[reverb_settings['genre'] == genre]
    ir_path = random.choice(genre_settings['file_path'].values)
    wet_dry_range = genre_settings.iloc[0][['wet_min', 'wet_max']].values
    print(f"Selected IR: {ir_path}")
    return ir_path, wet_dry_range

# Butterworth Filters
def butter_highpass(cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sos

def butter_lowpass(cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sos

def apply_filters(audio, sr, cutoff_freq):
    sos_high = butter_highpass(cutoff_freq, sr)
    sos_low = butter_lowpass(cutoff_freq, sr)
    high_frequencies = sosfilt(sos_high, audio)
    low_frequencies = sosfilt(sos_low, audio)
    return high_frequencies, low_frequencies

def apply_pre_delay(ir, pre_delay_ms, sr):
    pre_delay_samples = int(sr * pre_delay_ms / 1000)
    return np.pad(ir, (pre_delay_samples, 0), 'constant')

def apply_decay(ir, decay_time_ms, sr):
    decay_samples = int(sr * decay_time_ms / 1000)
    decay_curve = np.exp(-np.linspace(0, decay_time_ms, decay_samples) / decay_time_ms)
    decayed_ir = np.copy(ir)
    decayed_ir[:decay_samples] *= decay_curve
    return decayed_ir

def generate_early_reflections(input_signal, sr, reflection_times, gains):
    output_signal = np.zeros_like(input_signal)
    for time_ms, gain in zip(reflection_times, gains):
        delay_samples = int(sr * time_ms / 1000)
        delayed_signal = np.pad(input_signal, (delay_samples, 0), 'constant')[:len(input_signal)]
        output_signal += delayed_signal * gain
    return output_signal

def multitap_delay(input_signal, sr, tap_times, tap_gains, tap_feedbacks):
    output_signal = np.copy(input_signal)
    for time_ms, gain, feedback in zip(tap_times, tap_gains, tap_feedbacks):
        delay_samples = int(sr * time_ms / 1000)
        delayed_signal = np.pad(output_signal, (delay_samples, 0), 'constant')[:len(output_signal)]
        output_signal += delayed_signal * gain
        output_signal = output_signal * (1 - feedback)
    return output_signal

def adjust_room_size(ir, room_size_factor):
    # Placeholder for actual room size adjustment
    # Implementing this is complex and might involve convolution with room impulse responses
    # corresponding to different room sizes or advanced algorithmic techniques.
    return np.copy(ir)  # Placeholder implementation

def wet_dry_mix(dry_signal, wet_signal, wet_dry_ratio):
    mixed_signal = dry_signal * (1 - wet_dry_ratio) + wet_signal * wet_dry_ratio
    return mixed_signal

# Apply reverb to the high frequencies and mix with the dry low frequencies
# Modified apply_reverb function
def apply_reverb(audio, sr, ir_path, wet_dry_range, cutoff_freq, reverb_adjustments):
    high_frequencies, low_frequencies = apply_filters(audio, sr, cutoff_freq)

    # Load impulse response
    ir, _ = librosa.load(ir_path, sr=sr)

    # Apply reverb adjustments
    ir = apply_pre_delay(ir, reverb_adjustments['pre_delay'], sr)
    ir = apply_decay(ir, reverb_adjustments['decay_time'], sr)
    ir = adjust_room_size(ir, reverb_adjustments['room_size'])

    # Apply convolution to high frequencies
    reverbed_highs = fftconvolve(high_frequencies, ir, mode='full')[:len(high_frequencies)]

    # Apply wet/dry mix
    wet_dry_ratio = random.uniform(wet_dry_range[0], wet_dry_range[1])
    reverbed_highs *= wet_dry_ratio

    # Print statements after each adjustment
    print(f"Applying Pre-Delay: {reverb_adjustments['pre_delay']} ms")
    print(f"Applying Decay: {reverb_adjustments['decay_time']} ms")
    print(f"Adjusting Room Size: {reverb_adjustments['room_size']}")

    return wet_dry_mix(low_frequencies, reverbed_highs, wet_dry_ratio)

# Function to compute and return audio features
# Function to compute and return audio features
def compute_audio_features(audio, sr):
    # Spectral centroid and bandwidth for brightness and texture
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    print('Spectral Centroid: ', np.mean(spectral_centroid))
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    print('Spectral Bandwidth: ', np.mean(spectral_bandwidth))

    # Tempo and beat for rhythmic content analysis
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    print('Tempo: ', tempo)

    # Dynamic range for loudness variation
    rms = librosa.feature.rms(y=audio)
    dynamic_range = np.max(rms) - np.min(rms)
    print('Dynamic Range: ', dynamic_range)

    return {
        'spectral_centroid': np.mean(spectral_centroid),
        'spectral_bandwidth': np.mean(spectral_bandwidth),
        'tempo': tempo,
        'dynamic_range': dynamic_range
    }


# Function to adjust reverb parameters based on audio features
def adjust_reverb_parameters(features, base_wet_dry_ratio, reverb_settings):

    # Default reverb settings
    default_decay_time = 1.5  # Default decay time in seconds
    default_pre_delay = 20  # Default pre-delay in milliseconds
    default_room_size = 0.5  # Default room size (normalized scale)

    # Use values from reverb_settings if available, else use defaults
    decay_time = reverb_settings.get('decay_time', default_decay_time)
    pre_delay = reverb_settings.get('pre_delay', default_pre_delay)
    room_size = reverb_settings.get('room_size', default_room_size)

    # Print the original settings
    print("Original Reverb Settings:")
    print(
        f"Decay Time: {decay_time}, Pre-Delay: {pre_delay}, Room Size: {room_size}, Wet/Dry Ratio: {base_wet_dry_ratio}"
        )

    # Initialize adjustments with current or default settings
    reverb_adjustments = {
        'wet_dry_ratio': base_wet_dry_ratio,
        'decay_time': decay_time,
        'pre_delay': pre_delay,
        'room_size': room_size,
    }

    # Adjust based on brightness
    brightness_threshold = np.percentile(features['spectral_centroid'], 75)
    if features['spectral_centroid'] > brightness_threshold:
        reverb_adjustments['wet_dry_ratio'] *= 0.9

    # Adjust based on tempo
    if features['tempo'] > 120:
        reverb_adjustments['wet_dry_ratio'] *= 0.8
        reverb_adjustments['decay_time'] *= 0.9

    # Adjust based on spectral bandwidth
    if features['spectral_bandwidth'] > np.percentile(features['spectral_bandwidth'], 75):
        reverb_adjustments['room_size'] *= 1.1

    # Adjust based on dynamic range
    if features['dynamic_range'] > 0.1:
        reverb_adjustments['wet_dry_ratio'] *= 1.1
        reverb_adjustments['pre_delay'] *= 1.05

    # More complex conditions and adjustments can be added here using other audio features

    # Ensure that adjustments are within reasonable bounds
    reverb_adjustments['wet_dry_ratio'] = np.clip(reverb_adjustments['wet_dry_ratio'], 0, 1)
    reverb_adjustments['decay_time'] = max(reverb_adjustments['decay_time'], 0.1)  # for example
    reverb_adjustments['pre_delay'] = max(reverb_adjustments['pre_delay'], 0)  # for example
    reverb_adjustments['room_size'] = max(reverb_adjustments['room_size'], 0.1)  # for example

    # Print the adjusted settings
    print("Adjusted Reverb Settings:")
    print(
        f"Decay Time: {reverb_adjustments['decay_time']}, Pre-Delay: {reverb_adjustments['pre_delay']}, Room Size: {reverb_adjustments['room_size']}, Wet/Dry Ratio: {reverb_adjustments['wet_dry_ratio']}"
        )

    return reverb_adjustments



def process_audio_with_dynamic_reverb(input_file, output_file, genre):
    # Main process
    csv_path = 'reverb_settings.csv'  # Path to your CSV file
    print('Reading reverb settings...')
    reverb_settings = read_reverb_settings(csv_path)
    if genre not in reverb_settings['genre'].values:
        print('Genre not found in settings!')
        return
    print('Selecting random IR...')
    cutoff_freq = 250  # Cutoff frequency for filtering

    # Load audio
    audio, sr = librosa.load(input_file, sr=None)

    # Get reverb settings and audio features
    reverb_settings = read_reverb_settings('reverb_settings.csv')
    ir_path, wet_dry_range = select_random_ir(reverb_settings, genre)  # Get IR path and wet/dry range
    features = compute_audio_features(audio, sr)  # Compute audio features

    # Adjust reverb parameters based on audio content
    reverb_adjustments = adjust_reverb_parameters(features, wet_dry_range, reverb_settings)  # Adjust reverb settings
    reverbed_audio = apply_reverb(audio, sr, ir_path, wet_dry_range, cutoff_freq, reverb_adjustments )  # Apply reverb with adjustments
    # Save output
    sf.write(output_file, reverbed_audio, sr)


# Call the process function with an example file and genre
genre_selection = input("Enter genre: ")
process_audio_with_dynamic_reverb('SAMPLE.wav', 'output_reverbed_audio.wav', genre_selection)
print("Process completed. Audio with dynamic reverb applied.")
