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

    for file_path in tqdm.tqdm(file_paths, desc="Moving files"):
        try:
            shutil.move(file_path, target_dir)
            print(f"Moved '{file_path}' to '{target_dir}'")
        except Exception as e:
            print(f"Error moving '{file_path}': {e}")

if __name__ == "__main__":
    source_dir = r"D:\Data\Rubber"
    target_dir = r"D:\Data\temp"
    extensions = ['.zip', '.exe', '.gz', '.bz2', '.xz', '.7z', '.rar']

    find_files_by_extension(source_dir, target_dir, extensions)
    