import joblib
import pandas as pd
import ast
import os
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import accuracy_score
import numpy as np
import pefile  # Added import

import extract_to_load as etl


def load_model(model_dir):
    """Load all saved model components"""
    model_path = os.path.join(model_dir, "model_pipeline.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Load column names from CSV
    csv_path = os.path.join(model_dir, "template.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Template file not found: {csv_path}")
        
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    return model, columns


def extract_all_data(data_path):
    data_dict = {}

    try:
        pe = pefile.PE(data_path)  # Open PE file once
    except pefile.PEFormatError as e:
        print(f"Error parsing PE file: {e}")
        return None  # Return None or an empty dict if parsing fails

    # size of code extract
    size_of_code = etl.get_SizeOfCode(pe)  # Pass pe object
    if size_of_code is not None:
        data_dict['SizeOfCode'] = size_of_code

    # size of initialized data extract
    size_of_initialized_data = etl.get_SizeOfInitializedData(pe)  # Pass pe object
    if size_of_initialized_data is not None:
        data_dict['SizeOfInitializedData'] = size_of_initialized_data

    # size of image extract
    size_of_image = etl.get_SizeOfImage(pe)  # Pass pe object
    if size_of_image is not None:
        data_dict['SizeOfImage'] = size_of_image

    # subsystem extract
    subsystem = etl.get_Subsystem(pe)  # Pass pe object
    if subsystem is not None:
        data_dict['Subsystem'] = subsystem

    # get all dlls and their counts
    imported_dlls = etl.get_Imported_DLLs(pe)  # Pass pe object
    if imported_dlls is not None:
        for dll, functions in imported_dlls.items():
            temp = str(dll)
            temp = temp.replace(".dll", " ")
            temp = temp.strip()
            a = "dll_" + temp + "_dll_count"
            data_dict[a] = len(functions)  # Count the number of imported functions

    # get entropy of sections
    entropy_sections = etl.get_EntropyCalculation_and_sections(pe)  # Pass pe object
    if entropy_sections is not None:
        for section, entropy in entropy_sections.items():
            temp = str(section)
            cleaned_name = ''
            for char in str(temp):
                if char.isalnum():
                    cleaned_name += char
                else:
                    cleaned_name += '_'
            a = "entropy_" + cleaned_name + "_count"
            data_dict[a] = entropy

    pe.close()  # Close the PE file handle
    return data_dict


if __name__ == "__main__":
    # Load model
    print("-\t initializing tkinter\t-")
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    # model_dir = filedialog.askdirectory(title="Select Model Directory")
    
    print("-\t initializing model\t-")
    model_dir = r"C:\Users\bertr\Downloads\[SVM] trained_models(2025-04-24 20-09-03)"
    if not model_dir:
        print("No directory selected. Exiting.")
        exit(1)
    
    model, columns = load_model(model_dir)
    
    # Load new data
    data_path = filedialog.askopenfilename(title="Select Exe File", filetypes=[("Executable files", "*.exe;*.exev;*.exeh;*.exet")])
    if not data_path:
        print("No file selected. Exiting.")
        exit(1)
    
    active_data = extract_all_data(data_path)

    if active_data is None:  # Check if extraction failed
        print("Failed to extract data from the file. Exiting.")
        exit(1)
        
    # Initialize row_data with default values (0) for all expected columns
    row_data = {col: 0 for col in columns}
    
    # Update row_data with values from active_data if the column exists
    for col, value in active_data.items():
        if col in row_data:
            row_data[col] = value
            print(f"Mapped feature '{col}' with value: {value}")
        else:
            print(f"Warning: Extracted feature '{col}' not found in model's expected features")

    # Create DataFrame from the single row dictionary
    data_from_file = pd.DataFrame([row_data], columns=columns)
        
    # Convert categorical columns to strings
    categorical_features = ['Subsystem']
    for col in categorical_features:
        if col in data_from_file.columns:
            data_from_file[col] = data_from_file[col].astype(str)
    
    # Make prediction using the first row
    prediction = model.predict(data_from_file.iloc[0:1])
    probability = model.predict_proba(data_from_file.iloc[0:1])
    
    prob_clean = probability[0][0]
    prob_malware = probability[0][1]
    
    print("\nPrediction Probabilities:")
    print(f"Clean: {prob_clean:.2%}")
    print(f"Malware: {prob_malware:.2%}")
    
    is_malware = prediction[0] == 1
    certainty = 100 * (1 - prob_clean if is_malware else prob_clean)
    malware_certainty = 100 * (1 - prob_clean)
    
    print(f"Prediction: {'Malware' if is_malware else 'Clean'}")
    print(f"Certainty: {certainty:.2f}%")
    print(f"Malware Certainty: {malware_certainty:.2f}%")
