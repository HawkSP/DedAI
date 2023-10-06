import numpy as np
from scipy.integrate import odeint
import pandas as pd

# Time constants
DURATION = 20  # seconds
SAMPLE_RATE = 128  # Hz
TIME = np.arange(0, DURATION, 1 / SAMPLE_RATE)

# Music features (modifying to add complexity)
tempo = 105 + 15 * np.sin(2 * np.pi * 1 / 60 * TIME) * np.sin(2 * np.pi * 1 / 30 * TIME)
timbre = np.sin(2 * np.pi * 1 / 3 * TIME)
key = np.round(7 + 4 * np.sin(2 * np.pi * 1 / 10 * TIME) + 3 * np.sin(2 * np.pi * 1 / 40 * TIME))
genre = 4 + 2 * np.sin(2 * np.pi * 1 / 30 * TIME) + np.sin(2 * np.pi * 1 / 60 * TIME)

# Influence matrix for emotions
emotion_influence_matrix = np.array([
    [0.1, 0.05, -0.05, 0.05, -0.1, 0.02, -0.01],
    [0.05, 0.1, -0.05, 0.1, -0.05, 0.01, -0.02],
    [-0.05, -0.05, 0.1, -0.1, 0.05, 0.02, -0.01],
    [0.05, 0.1, -0.1, 0.1, -0.05, -0.01, 0.02],
    [-0.1, -0.05, 0.05, -0.05, 0.1, 0.01, 0.01],
    [0.02, 0.01, -0.01, 0.02, 0.01, 0.1, -0.05],
    [-0.01, -0.02, 0.01, -0.02, 0.01, 0.05, 0.1]
])


# Influence matrix for musical features
musical_feature_influence_matrix = np.array([
    [0.02, 0.01, -0.02, 0.01],
    [-0.01, 0.02, 0.01, -0.02],
    [0.02, -0.01, 0.02, 0.01],
    [-0.01, 0.01, -0.02, 0.02]
])


# Sigmoid function to model neural activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Differential equations for EEG emotions
def emotion_system(y, t):
    excitement, long_term_excitement, frustration, engagement, relaxation, interest_affinity, focus = y
    emotion_values = np.array(y[:7])

    musical_feature_values = np.array([tempo[i], timbre[i], key[i], genre[i]])

    # Apply influence
    emotion_changes = emotion_influence_matrix.dot(emotion_values)
    musical_feature_changes = musical_feature_influence_matrix.dot(musical_feature_values)

    # Change in Excitement
    d_excitement = 0.1 * engagement - 0.05 * frustration + 0.1 * np.sin(2 * np.pi * 1 / 60 * t) + emotion_changes[0]

    # Change in Long-Term Excitement
    d_long_term_excitement = 0.05 * excitement + 0.03 * engagement - 0.01 * long_term_excitement + emotion_changes[1]

    # Change in Frustration
    d_frustration = 0.05 * (
                1 - engagement) + 0.03 * long_term_excitement - 0.02 * relaxation - 0.03 * excitement + 0.01 * abs(
        tempo[i] - tempo[i - 1]) - 0.02 * frustration + emotion_changes[2]

    # Change in Engagement
    d_engagement = 0.1 * excitement - 0.05 * engagement + 0.03 * focus + emotion_changes[3]

    # Change in Relaxation
    d_relaxation = -0.2 * frustration + 0.3 * relaxation + 0.1 * interest_affinity - 0.05 * excitement + emotion_changes[4]

    # Change in Interest Affinity
    d_interest_affinity = 0.05 * focus + 0.03 * relaxation - 0.02 * interest_affinity + emotion_changes[5]

    # Change in Focus
    d_focus = 0.1 * engagement - 0.05 * focus - 0.03 * long_term_excitement + 0.02 * tempo[i] + emotion_changes[6]

    return [d_excitement, d_long_term_excitement, d_frustration, d_engagement, d_relaxation, d_interest_affinity, d_focus]

# Initial conditions
initial_conditions = [25, 25, 25, 25, 25, 25, 25]

# Solve ODE with changing multipliers
refined_eeg_data = []
for i in range(len(TIME) - 1):
    # Solve ODE for this step
    result = odeint(emotion_system, initial_conditions, [TIME[i], TIME[i + 1]])

    refined_eeg_data.append(result[0])
    initial_conditions = result[-1]

refined_eeg_data = np.array(refined_eeg_data)

# Scaling the EEG data
refined_eeg_data = (refined_eeg_data - refined_eeg_data.min(axis=0)) / (
            refined_eeg_data.max(axis=0) - refined_eeg_data.min(axis=0)) * 100

# Create DataFrame
refined_data = pd.DataFrame({
    'time': TIME[:-1],
    'tempo': tempo[:-1],
    'timbre': timbre[:-1],
    'key': key[:-1],
    'genre': genre[:-1],
    'excitement': refined_eeg_data[:, 0],
    'long_term_excitement': refined_eeg_data[:, 1],
    'frustration': refined_eeg_data[:, 2],
    'engagement': refined_eeg_data[:, 3],
    'relaxation': refined_eeg_data[:, 4],
    'interest_affinity': refined_eeg_data[:, 5],
    'focus': refined_eeg_data[:, 6]
})

# Save to CSV file
refined_data_path = 'dataset2.csv'
refined_data.to_csv(refined_data_path, index=False)
