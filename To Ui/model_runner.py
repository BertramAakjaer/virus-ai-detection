import joblib
import pandas as pd
import os
import pefile

import extract_to_load as etl

MODEL = None
COLUMNS = None


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

    # get all dlls and their functions
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


def init_model(model_dir):
    """Initialize the model and load the data"""
    try:
        global MODEL, COLUMNS
        
        if not os.path.exists(model_dir):
            return None

        model, columns = load_model(model_dir)
        
        MODEL = model
        COLUMNS = columns
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def run_model(file_path):
    """Run the model on the selected data file"""
    if not file_path:
        print("No file selected. Exiting.")
        exit(1)
    
    active_data = extract_all_data(file_path)

    if active_data is None:  # Check if extraction failed
        print("Failed to extract data from the file. Exiting.")
        exit(1)
        
    # Initialize row_data with default values (0) for all expected columns
    row_data = {col: 0 for col in COLUMNS}
    
    # Update row_data with values from active_data if the column exists
    for col, value in active_data.items():
        if col in row_data:
            row_data[col] = value
        else:
            print(f"Warning: Extracted feature '{col}' not found in model's expected features")

    # Create DataFrame from the single row dictionary
    data_from_file = pd.DataFrame([row_data], columns=COLUMNS)
    
    missing_features = set(COLUMNS) - set(active_data.keys())

    
    # Convert categorical columns to strings to match training data format
    categorical_features = ['Subsystem']  # Add other categorical features if any
    for col in categorical_features:
        if col in data_from_file.columns:
            data_from_file[col] = data_from_file[col].astype(str)
    
    # Make prediction using the first row
    prediction = MODEL.predict(data_from_file.iloc[0:1])
    probability = MODEL.predict_proba(data_from_file.iloc[0:1])
    
    # For consistency, all models return probabilities where:
    # - index 0 is the probability of being clean (class 0)
    # - index 1 is the probability of being malware (class 1)
    prob_clean = probability[0][0]
    prob_malware = probability[0][1]

    
    is_malware = prediction[0] == 1
    # If it's malware, return the malware probability as certainty
    # If it's clean, return the clean probability as certainty
    certainty = 100 * (1 - prob_clean if is_malware else prob_clean)
    malware_certainty = 100 * (1 - prob_clean)
    
    return is_malware, certainty, malware_certainty
