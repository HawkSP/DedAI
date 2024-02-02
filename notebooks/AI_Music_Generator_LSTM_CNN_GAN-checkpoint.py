# Importing necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
import librosa
import pickle
from skimage.transform import resize

print("Libraries imported successfully!")

print("Adding functions...")


# Function to walk through a directory and collect WAV files
def collect_audio_files(directory):
	audio_files = []
	for root, dirs, files in os.walk(directory):
		for file in files:
			if file.endswith('.wav'):
				audio_files.append(os.path.join(root, file))
	return audio_files


# Modify the preprocess_and_store_data function to use resize_spectrogram
def preprocess_and_store_data(audio_files, feature_store_path):
	spectrograms = []
	target_shape = (128, 128)  # Desired shape for CNN input

	for file in audio_files:
		audio, sr = librosa.load(file, sr=None)
		# Ensure correct usage of melspectrogram function
		spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
		spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
		spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)

		# Resize spectrogram to target_shape (128x128)
		spectrogram = resize_spectrogram(spectrogram, target_shape)

		# Add a channel dimension
		spectrogram = spectrogram[..., np.newaxis]

		spectrograms.append(spectrogram)

	# Store the processed data
	with open(feature_store_path, 'wb') as f:
		pickle.dump(spectrograms, f)

	return np.array(spectrograms)


def resize_spectrogram(spectrogram, target_shape):
	"""
    Resize the spectrogram to the target shape.
    This function will crop or pad the spectrogram as necessary.

    :param spectrogram: 2D array of the spectrogram
    :param target_shape: Tuple (height, width) for the target shape
    :return: Resized spectrogram
    """
	current_shape = spectrogram.shape

	# Padding if necessary
	if current_shape[0] < target_shape[0] or current_shape[1] < target_shape[1]:
		padding = [(0, max(0, target_shape[0] - current_shape[0])),
		           (0, max(0, target_shape[1] - current_shape[1]))]
		spectrogram = np.pad(spectrogram, padding, mode='constant', constant_values=0)

	# Cropping if necessary
	cropped_spectrogram = spectrogram[:target_shape[0], :target_shape[1]]

	return cropped_spectrogram


# Load preprocessed data from storage
def load_preprocessed_data(feature_store_path):
	with open(feature_store_path, 'rb') as f:
		spectrograms = pickle.load(f)
	return np.array(spectrograms)


# CNN for Feature Extraction
def create_cnn():
	input_shape = (128, 128, 1)
	inputs = Input(shape=input_shape)
	x = Conv2D(32, (3, 3), activation='relu')(inputs)
	x = MaxPooling2D((2, 2))(x)
	x = Conv2D(64, (3, 3), activation='relu')(x)
	x = MaxPooling2D((2, 2))(x)
	x = Flatten()(x)
	model = Model(inputs, x)
	return model


# LSTM for Temporal Dynamics
def create_lstm(cnn_output_size):
	inputs = Input(shape=(None, cnn_output_size))
	x = LSTM(256, return_sequences=False)(inputs)  # Removed one LSTM layer for simplicity
	x = Dense(128 * 128, activation='relu')(x)  # Reshape to 128*128
	x = Reshape((128, 128, 1))(x)  # Reshape to match CNN input
	model = Model(inputs, x)
	return model


# GAN for Music Generation
def create_gan(cnn, lstm):
	generator = lstm
	discriminator = cnn
	discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

	# Generator
	z = Input(shape=(None, cnn.output_shape[-1]))
	generated_spectrogram = generator(z)

	# Discriminator
	discriminator.trainable = False
	validity = discriminator(generated_spectrogram)

	combined = Model(z, validity)
	combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
	return combined


# Main Training Loop
def train_gan(combined, generator, discriminator, spectrograms, epochs, batch_size):
	for epoch in range(epochs):
		idx = np.random.randint(0, spectrograms.shape[0], batch_size)
		real_spectrograms = spectrograms[idx]
		noise = np.random.normal(0, 1, (batch_size, 100))
		generated_spectrograms = generator.predict(noise)
		real_loss = discriminator.train_on_batch(real_spectrograms, np.ones((batch_size, 1)))
		fake_loss = discriminator.train_on_batch(generated_spectrograms, np.zeros((batch_size, 1)))
		discriminator_loss = 0.5 * np.add(real_loss, fake_loss)
		noise = np.random.normal(0, 1, (batch_size, 100))
		generator_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))
		print(f"Epoch {epoch} / {epochs} - Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}")


# Music Generation Function
def generate_music(generator, num_samples, timesteps=1):
	# Generate random noise
	noise = np.random.normal(0, 1, (num_samples, 100))

	# Reshape noise to add timesteps dimension
	noise = np.expand_dims(noise, axis=1)  # New shape: (num_samples, 1, 100)

	# Predict using the generator
	generated_spectrograms = generator.predict(noise)
	return generated_spectrograms


# AWS Integration Placeholder (Dormant)
def aws_integration_placeholder():
	pass


print("Functions added successfully!")

# Directory containing WAV files
directory = 'datasets/nsynth/audio'
audio_files = collect_audio_files(directory)  #
print(f"Found {len(audio_files)} audio files")

# Path to store extracted features
feature_store_path = 'datasets/nsynth/store.pkl'
if os.path.exists(feature_store_path):
	print(f"Feature store path: {feature_store_path}")
	os.makedirs(os.path.dirname(feature_store_path), exist_ok=True)
else:
	print(f"Feature store path does not exist: {feature_store_path}")
	print(f"Creating feature store path: {feature_store_path}")

os.makedirs(os.path.dirname(feature_store_path), exist_ok=True)

# Check if preprocessed data exists
if os.path.exists(feature_store_path):
	spectrograms = load_preprocessed_data(feature_store_path)
	print(f"Loaded {len(spectrograms)} spectrograms")
else:
	spectrograms = preprocess_and_store_data(audio_files, feature_store_path)
	print(f"Preprocessed and stored {len(spectrograms)} spectrograms")

# Prepare models
cnn = create_cnn()
print(cnn.summary())
lstm = create_lstm(cnn.output_shape[-1])
print(lstm.summary())
gan = create_gan(cnn, lstm)
print(gan.summary())

# Train the GAN
train_gan(gan, lstm, cnn, spectrograms, epochs=1000, batch_size=32)
print("GAN training complete!")

# Generate music
generated_music = generate_music(lstm, 5)  # Generate 5 samples
print("Generated Music Shape:", generated_music.shape)

# Additional processing to convert spectrograms back to audio
