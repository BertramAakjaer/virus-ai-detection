import os
import hashlib
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import time

def remove_duplicate_files(folder_path):
    """Removes duplicate files in a folder using MD5 hash."""

    hashes = {}
    duplicates = []

    file_list = [filename for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]
    
    for filename in tqdm(file_list, desc="Checking files", unit="file"):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        if file_hash in hashes:
            duplicates.append(file_path)
            print(f"Duplicate found: {filename} (same as {hashes[file_hash]})")
        else:
            hashes[file_hash] = filename

    for file_path in duplicates:
        os.remove(file_path)
        print(f"Removed: {file_path}")
    
    print(f"Removed {len(duplicates)} duplicate files.")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    folder_path = filedialog.askdirectory()

    if folder_path:
        remove_duplicate_files(folder_path)
    else:
        print("No folder selected.")