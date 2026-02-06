import os
import shutil
import random

# --- Configuration ---
SOURCE_DIR = "screenshots"          # Folder containing all screenshots (0_*.png, 1_*.png)
DEST_DIR = "cv_data_train"          # Main folder for training data
LEGIT_SUBDIR = os.path.join(DEST_DIR, "legitimate") # Subfolder for legitimate images
PHISH_SUBDIR = os.path.join(DEST_DIR, "phishing")   # Subfolder for phishing images
NUM_SAMPLES_PER_CLASS = 500         # How many images to copy for each class
# ---------------------

print("--- Preparing CV Training Data Subset ---")

# 1. Create destination directories if they don't exist
os.makedirs(LEGIT_SUBDIR, exist_ok=True)
os.makedirs(PHISH_SUBDIR, exist_ok=True)
print(f"Ensured directories exist: '{LEGIT_SUBDIR}' and '{PHISH_SUBDIR}'")

# 2. List all files in the source directory
try:
    all_files = [f for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f)) and f.endswith('.png')]
    print(f"Found {len(all_files)} total PNG files in '{SOURCE_DIR}'.")
except FileNotFoundError:
    print(f"[ERROR] Source directory '{SOURCE_DIR}' not found. Did Phase 1 run correctly?")
    exit()

# 3. Filter files by prefix
legit_files = [f for f in all_files if f.startswith('0_')]
phish_files = [f for f in all_files if f.startswith('1_')]
print(f"Found {len(legit_files)} legitimate files (starting with '0_').")
print(f"Found {len(phish_files)} phishing files (starting with '1_').")

# 4. Select a random subset
num_legit_to_copy = min(NUM_SAMPLES_PER_CLASS, len(legit_files))
num_phish_to_copy = min(NUM_SAMPLES_PER_CLASS, len(phish_files))

selected_legit = random.sample(legit_files, num_legit_to_copy)
selected_phish = random.sample(phish_files, num_phish_to_copy)

print(f"\nSelected {num_legit_to_copy} legitimate files to copy.")
print(f"Selected {num_phish_to_copy} phishing files to copy.")

# 5. Copy the selected files
copied_legit_count = 0
print(f"\nCopying legitimate files to '{LEGIT_SUBDIR}'...")
for filename in selected_legit:
    source_path = os.path.join(SOURCE_DIR, filename)
    dest_path = os.path.join(LEGIT_SUBDIR, filename)
    try:
        shutil.copy2(source_path, dest_path) # copy2 preserves metadata
        copied_legit_count += 1
    except Exception as e:
        print(f"  [Error] Could not copy {filename}: {e}")
print(f"Copied {copied_legit_count} legitimate files.")

copied_phish_count = 0
print(f"\nCopying phishing files to '{PHISH_SUBDIR}'...")
for filename in selected_phish:
    source_path = os.path.join(SOURCE_DIR, filename)
    dest_path = os.path.join(PHISH_SUBDIR, filename)
    try:
        shutil.copy2(source_path, dest_path)
        copied_phish_count += 1
    except Exception as e:
        print(f"  [Error] Could not copy {filename}: {e}")
print(f"Copied {copied_phish_count} phishing files.")

print("\n--- CV Data Preparation Finished ---")