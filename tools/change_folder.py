import os
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

EXTENTION = '.exeh'
MAX_FILENAME_LENGTH = 255  # Maximum filename length (OS dependent)

def change_file_extensions(folder_path):
    try:
        # Get the list of files in the folder
        files = os.listdir(folder_path)
        
        # Use tqdm to create a progress bar
        for filename in tqdm(files, desc="Processing files"):
            file_path = os.path.join(folder_path, filename)
            
            # Check if it's a file
            if os.path.isfile(file_path):
                # Split the file name and extension
                base_name, ext = os.path.splitext(filename)
                
                # Skip if the file already has the desired extension
                if ext == EXTENTION:
                    #print(f"Skipping: {file_path} already has the correct extension.")
                    continue
                
                # Truncate the base name if it's too long
                if len(filename + EXTENTION) > MAX_FILENAME_LENGTH:
                    base_name = filename[:MAX_FILENAME_LENGTH - len(EXTENTION)]
                
                # Create the new file name with .exeh extension
                new_file_path = os.path.join(folder_path, filename + EXTENTION)
                # Rename the file
                try:
                    os.rename(file_path, new_file_path)
                    #print(f"Renamed: {file_path} -> {new_file_path}")
                except FileExistsError:
                    #print("File exists, trying to overwrite")
                    os.remove(new_file_path)
                    os.rename(file_path, new_file_path)
                    #print(f"Renamed (after overwrite): {file_path} -> {new_file_path}")
                except Exception as e:
                    print(f"An error occurred while renaming {file_path}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage with tkinter file explorer

root = tk.Tk()
root.withdraw()  # Hide the root window
folder_to_process = filedialog.askdirectory(title="Select Folder")
change_file_extensions(folder_to_process)