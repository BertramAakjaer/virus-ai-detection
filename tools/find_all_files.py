import os
import shutil
import argparse
import tqdm

def find_files_by_extension(source_dir, target_dir, extensions):
    file_paths = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    errors = 0
    skipped = 0
    copied = 0

    for file_path in tqdm.tqdm(file_paths, desc="Moving files"):
        target_path = os.path.join(target_dir, os.path.basename(file_path))
        if os.path.exists(target_path):
            skipped += 1
            continue
        try:
            shutil.move(file_path, target_dir)
            copied += 1
        except Exception as e:
            print(f"Error moving '{file_path}': {e}")
            errors += 1
    
    print(f"Files copied: {copied}, Files skipped: {skipped}, Errors: {errors}")

if __name__ == "__main__":
    source_dir = r"C:\Users\bertr\Downloads\winpe"
    target_dir = r"D:\Data\temp"
    extensions = ['.zip', '.exe', '.gz', '.bz2', '.xz', '.7z', '.rar']

    find_files_by_extension(source_dir, target_dir, extensions)