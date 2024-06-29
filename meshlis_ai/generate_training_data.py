import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the positions of the microphones
mic_positions = np.array([
    [0, 1],  # Mic 1 at (0, 1)
    [np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)],  # Mic 2 at 120 degrees to bottom left
    [np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3)]   # Mic 3 at 120 degrees to bottom right
])

def calculate_loudness(source_pos, mic_pos, initial_loudness, noise_std=0.1):
    distance = np.linalg.norm(source_pos - mic_pos)
    distance = distance * 50
    
    # Calculate sound pressure level (SPL) using inverse square law
    if distance > 0:
        spl = initial_loudness - 20 * np.log10(distance)
    else:
        spl = initial_loudness

    # Add Gaussian noise
    noise = np.random.normal(0, noise_std)
    spl += noise
    if spl < 0:
        spl = 0
    return spl

# Generate training data
data = []
for x in np.linspace(-2, 2, 1000):
    for y in np.linspace(-2, 2, 1000):
        source_pos = np.array([x, y])
        initial_loudness = np.random.uniform(40, 100)
        loudness_values = [calculate_loudness(source_pos, mic_pos, initial_loudness) for mic_pos in mic_positions]
        data.append(loudness_values + [x, y])

# Convert to DataFrame
columns = ['mic1_loudness', 'mic2_loudness', 'mic3_loudness', 'x', 'y']
df = pd.DataFrame(data, columns=columns)

# Shuffle the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV (optional)
df.to_csv('sound_source_data_with_noise.csv', index=False)