import joblib
import pandas as pd
import ast
import os
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import accuracy_score
import numpy as np

import extract_to_load as etl


def load_model(model_dir):
    """Load all saved model components"""
    model = joblib.load(os.path.join(model_dir, "model_pipeline.pkl"))
    
    # Load column names from CSV
    csv_path = os.path.join(model_dir, "template.csv")
    if csv_path:
        columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    return model, columns


def extract_all_data(data_path):
    data_dict = {}
    
    # size of code extract
    size_of_code = etl.get_SizeOfCode(data_path)
    if size_of_code is not None:
        data_dict['SizeOfCode'] = size_of_code
        
    # size of initialized data extract
    size_of_initialized_data = etl.get_SizeOfInitializedData(data_path)
    if size_of_initialized_data is not None:
        data_dict['SizeOfInitializedData'] = size_of_initialized_data
    
    # size of image extract
    size_of_image = etl.get_SizeOfImage(data_path)
    if size_of_image is not None:
        data_dict['SizeOfImage'] = size_of_image
    
    # subsystem extract
    subsystem = etl.get_Subsystem(data_path)
    if subsystem is not None:
        data_dict['Subsystem'] = subsystem
    
    # get all dlls and their counts
    imported_dlls = etl.get_Imported_DLLs(data_path)
    if imported_dlls is not None:
        for dll, count in imported_dlls.items():
            temp = str(dll)
            
                        
            temp = temp.replace(".dll", " ")
            temp = temp.strip()
            
            a = "dll_" + temp + "_dll_count"
            
            data_dict[a] = count
    
    

    # get entropy of sections
    entropy_sections = etl.get_EntropyCalculation_and_sections(data_path)
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
        
    return data_dict
        
if __name__ == "__main__":
    # Load model
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    #model_dir = filedialog.askdirectory(title="Select Model Directory")
    model_dir = r"C:\Users\bertr\OneDrive - NEXT Uddannelse København\Skrivebord\virus-ai-detection\AI training\trained_models(2025-04-23 11-53-40)"
    if not model_dir:
        print("No directory selected. Exiting.")
        exit(1)
    
    model, columns = load_model(model_dir)
    
    # Load new data
    data_path = filedialog.askopenfilename(title="Select Exe File", filetypes=[("Executable files", "*.exe;*.exev;*.exeh;*.exet")])
    
    if not data_path:
        print("No file selected. Exiting.")
        exit(1)
    
    data_from_file = pd.DataFrame(columns=columns)
    active_data = extract_all_data(data_path)
    
    print(f"Extracted data: {active_data}")
    
    for col in columns:
        if col in active_data:
            data_from_file[col] = [active_data[col]]
            print(f"Column '{col}' found in data: {active_data[col]}")
        else:
            data_from_file[col] = [0]
        
    
    # Create empty DataFrame with specified columns
    
    #print(data_from_file)
    
    # Export to Excel
    #output_path = "extracted_features.xlsx"
    #data_from_file.to_excel(output_path, index=False)
    #print(f"\nFeatures exported to: {output_path}")
    
    # Make prediction using the first row
    prediction = model.predict(data_from_file.iloc[0:1])

    scores = model.decision_function(data_from_file.iloc[0:1])
    probability = [[1 / (1 + np.exp(-x)) for x in scores]]
    
    prob_procent = 100 * probability[0][0]

    # print(prediction)
    # print(probability)
    print("\n")
    
    print("\nPrediction:", "Malware" if prediction[0] == 0 else "Clean")
    print("Probability: {:.2f} %".format(prob_procent if prediction[0] == "malware" else 100 - prob_procent))
