import os
import tkinter as tk
from tkinter import filedialog

EXTENTION = '.exev'
MAX_FILENAME_LENGTH = 255  # Maximum filename length (OS dependent)

def change_file_extensions(folder_path):
    try:
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Check if it's a file
            if os.path.isfile(file_path):
                # Split the file name and extension
                base_name, _ = os.path.splitext(filename)
                
                # Truncate the base name if it's too long
                if len(base_name + EXTENTION) > MAX_FILENAME_LENGTH:
                    base_name = base_name[:MAX_FILENAME_LENGTH - len(EXTENTION)]
                
                # Create the new file name with .exeh extension
                new_file_path = os.path.join(folder_path, base_name + EXTENTION)
                # Rename the file
                try:
                    os.rename(file_path, new_file_path)
                    print(f"Renamed: {file_path} -> {new_file_path}")
                except FileExistsError:
                    os.remove(new_file_path)
                    os.rename(file_path, new_file_path)
                    print(f"Renamed (after overwrite): {file_path} -> {new_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage with tkinter file explorer

root = tk.Tk()
root.withdraw()  # Hide the root window
folder_to_process = filedialog.askdirectory(title="Select Folder")
change_file_extensions(folder_to_process)