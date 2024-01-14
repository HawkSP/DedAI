import os
import shutil

# Define the root directory where the IR folders are located
root_dir = 'IR'
destination_dir = 'IR_export'

# Function to copy WAV files from 'stereo' subdirectories to a destination directory
def collect_ir_files(root, dest):
    for subdir, dirs, files in os.walk(root):
        # Check if the current directory is 'stereo'
        if os.path.basename(subdir) == 'stereo':
            for file in files:
                # Check if the file is a WAV file
                if file.lower().endswith('.wav'):
                    # Construct the full file path
                    file_path = os.path.join(subdir, file)
                    # Copy the file to the destination directory
                    shutil.copy(file_path, dest)
                    print(f"Copied {file} to {dest}")


# Calling the function to start the process
collect_ir_files(root_dir, destination_dir)
