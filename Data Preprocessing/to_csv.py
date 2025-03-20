import tkinter as tk
from tkinter import filedialog
import pandas as pd
from tqdm import tqdm
import pefile, os

import extract_from_exe as efe



VIRUS_EXTENTION = '.exev'
HARMLESS_EXTENTION = '.exeh'



def init_table():
    return pd.DataFrame(columns=['Name', 'SizeOfCode', 'SizeOfInitializedData', 'SizeOfImage', 'Subsystem', 'EntropyAndSections', 'ImportetDLLS', 'Label'])

def add_row(df, path):
    pe = pefile.PE(path)

    name                        =   os.path.basename(path)
    size_of_code                =   efe.get_SizeOfCode(pe)
    size_of_initialized_data    =   efe.get_SizeOfInitializedData(pe)
    size_of_image               =   efe.get_SizeOfImage(pe)
    subsystem                   =   efe.get_Subsystem(pe)
    entropes_and_sections       =   efe.get_EntropyCalculation_and_sections(pe)
    importet_dlls               =   efe.get_Importet_DLLs(pe)

    label = None

    if path.lower().endswith(VIRUS_EXTENTION):
        label = "malware"
    elif path.lower().endswith(HARMLESS_EXTENTION):
        label = "harmless"
    else:
        print(f"Unknown file type: {path}")
        label = "unknown"

    df.loc[len(df)] = [name, size_of_code, size_of_initialized_data, size_of_image, subsystem, entropes_and_sections, importet_dlls, label]

def extract_files(df, folder_path):
    files_to_process = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((VIRUS_EXTENTION, HARMLESS_EXTENTION)):
                files_to_process.append(os.path.join(root, file))
    
    for file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
        add_row(df, file_path)
    
    return df

if __name__ == '__main__':

    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory(title="Select Folder")

    print(f"Selected folder: {folder_path}")

    df = init_table()

    df = extract_files(df, folder_path)

    
    df.to_csv('data.csv', index=False)