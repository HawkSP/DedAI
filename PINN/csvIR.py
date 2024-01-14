import os
import pandas as pd
import random

# Define the directory containing the IR files
ir_export_dir = 'IR_export'  # Replace with the path to your IR_export directory

# Define your genre to wet/dry mapping
genre_wet_dry = {
        'Pop': (0.2, 0.4),
        'Rock': (0.3, 0.5),
        'Classical': (0.4, 0.6),
        'Jazz': (0.25, 0.45),
        'Blues': (0.3, 0.5),
        'Country': (0.15, 0.35),
        'Hip Hop': (0.2, 0.4),
        'Electronic': (0.5, 0.7),
        'Reggae': (0.4, 0.6),
        'Folk': (0.2, 0.4),
        'Soul': (0.3, 0.5),
        'R&B': (0.3, 0.5),
        'Metal': (0.3, 0.5),
        'Punk': (0.2, 0.4),
        'Indie': (0.25, 0.45),
        'Latin': (0.4, 0.6),
        'Opera': (0.5, 0.7),
        'Reggaeton': (0.3, 0.5),
        'Gospel': (0.3, 0.5),
        'EDM': (0.6, 0.8),
        'Jazz Fusion': (0.25, 0.45),
        'Ska': (0.2, 0.4),
        'Funk': (0.35, 0.55),
        'Ambient': (0.1, 0.3),
        'World Music': (0.3, 0.5)
    }

# List to hold all IR file info
ir_files_info = []


# Function to get random wet/dry values based on genre
def get_wet_dry_values(genre):
    return random.uniform(*genre_wet_dry[genre])


# Walk through the directory and get all '.wav' files
for subdir, dirs, files in os.walk(ir_export_dir):
    print(f"Checking in directory: {subdir}")  # Print the current directory
    for file in files:
        print(f"Found file: {file}")  # Print each file found
        if file.endswith('.wav'):
            print(f"Processing file: {file}")  # Print the file being processed
            # Avoiding adding repeat genres
            available_genres = list(set(genre_wet_dry.keys()) - set([info['genre'] for info in ir_files_info]))
            if not available_genres:  # If all genres are used, reset the available genres
                available_genres = list(genre_wet_dry.keys())
            selected_genre = random.choice(available_genres)
            wet_min, wet_max = genre_wet_dry[selected_genre]
            ir_files_info.append({
                'file_path': os.path.join(subdir, file),
                'genre': selected_genre,
                'wet_min': wet_min,
                'wet_max': wet_max
            })
            print(f"Added file to list with genre: {selected_genre}")  # Confirm file is added

# Create a DataFrame from the list of dicts
reverb_settings_df = pd.DataFrame.from_records(ir_files_info)

# Write the DataFrame to a CSV file
csv_path = 'reverb_settings.csv'  # Change to your desired CSV file path
reverb_settings_df.to_csv(csv_path, index=False)

print(f"CSV file created with {len(reverb_settings_df)} entries at {csv_path}")
