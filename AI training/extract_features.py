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
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    print("-\t initializing model\t-")
    model_dir = r"C:\Users\bertr\Downloads\drive-download-20250425T075911Z-001\Big Dataset (70 MB)\[SVM] trained_models(2025-04-25 09-49-54)"
    if not model_dir:
        print("No directory selected. Exiting.")
        exit(1)
    
    # Load the model and template
    model, columns = load_model(model_dir)
    
    print("-\t Getting SVM weights\t-")
    # Extract the SVM model (the last step) and preprocessor
    svm_model = model.steps[-1][1]
    preprocessor = model.steps[0][1]
    
    # Get the weights directly (they're already transposed correctly in coef_)
    weights = svm_model.coef_[0]
    
    # Create feature names list
    feature_names = []
    
    # Add all non-Subsystem columns first
    numeric_features = [col for col in columns if col != 'Subsystem']
    feature_names.extend(numeric_features)
    
    # Get the OneHotEncoder from the categorical transformer pipeline
    if 'cat' in preprocessor.named_transformers_:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        if hasattr(cat_encoder, 'categories_'):
            # Add one-hot encoded features for Subsystem
            for cat in cat_encoder.categories_[0]:
                feature_names.append(f'Subsystem_{cat}')
    
    # Match feature names length with weights length
    if len(feature_names) > len(weights):
        feature_names = feature_names[:len(weights)]
    elif len(feature_names) < len(weights):
        feature_names.extend([f'Unknown_Feature_{i}' for i in range(len(feature_names), len(weights))])
    
    # Create DataFrame with weights
    weights_df = pd.DataFrame({
        'Feature': feature_names,
        'Feature Importance': weights,
        'Absolute Importance': abs(weights)
    })
    
    # Sort by absolute importance
    weights_df = weights_df.sort_values('Absolute Importance', ascending=False)
    
    # Create output filename with timestamp
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = os.path.join(model_dir, f'feature_weights_{timestamp}.xlsx')
    
    # Save to Excel file
    weights_df.to_excel(output_file, index=False, sheet_name='Feature Weights')
    
    print(f"\nFeature weights have been saved to: {output_file}")
    print("\nTop 10 most important features:")
    print(weights_df.head(10))
