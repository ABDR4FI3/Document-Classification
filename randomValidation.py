import os
import random
import shutil

# Source and destination paths
source_dir = r"C:\Users\Elgue\Documents\Last Semester\DL OCR\dataset"
destination_dir = r"C:\Users\Elgue\Documents\Last Semester\DL OCR\validation"

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Collect all file paths from the source directory and its subdirectories
all_files = []
for root, _, files in os.walk(source_dir):
    for file in files:
        # Check if the file is an image (you can add more extensions if needed)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            all_files.append(os.path.join(root, file))

# Select 50 random files (or fewer if there aren't 50 files available)
selected_files = random.sample(all_files, min(50, len(all_files)))

# Copy the selected files to the destination directory
for file_path in selected_files:
    shutil.copy(file_path, destination_dir)

print(f"Copied {len(selected_files)} images to {destination_dir}")
