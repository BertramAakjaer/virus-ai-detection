import joblib
import pandas as pd
import ast
import os
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import accuracy_score

def load_model(model_dir):
    """Load all saved model components"""
    model = joblib.load(os.path.join(model_dir, "model_pipeline.pkl"))
    dlls = joblib.load(os.path.join(model_dir, "dlls.pkl"))
    entropies = joblib.load(os.path.join(model_dir, "entropies.pkl"))
    feature_info = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
    return model, dlls, entropies, feature_info

def process_data(data, dlls, entropies, feature_info):
    """Process new data using same steps as training"""
    data = data.dropna()
    
    # Convert string representations to dictionaries
    data['EntropyAndSections'] = data['EntropyAndSections'].apply(ast.literal_eval)
    data['ImportedDLLS'] = data['ImportedDLLS'].apply(ast.literal_eval)
    
    # Create DLL features using only the DLLs from training
    dll_counts = {}
    for dll in sorted(dlls):  # Sort to ensure consistent order
        dll_counts[f'dll_{dll}_count'] = data['ImportedDLLS'].apply(lambda x: len(x.get(dll, [])))
    dll_df = pd.DataFrame(dll_counts)

    # Create entropy features using only the entropies from training
    entropy_counts = {}
    for entropy in sorted(entropies):  # Sort to ensure consistent order
        entropy_counts[f'entropy_{entropy}_count'] = data['EntropyAndSections'].apply(lambda x: len(str(x.get(entropy, []))))
    entropy_df = pd.DataFrame(entropy_counts)

    # Combine all features like in training
    data = pd.concat([data, dll_df, entropy_df], axis=1)

    # Prepare features matching training
    feature_df = data.drop(columns=['EntropyAndSections', 'ImportedDLLS', 'Name'])
    if 'Label' in feature_df.columns:
        feature_df = feature_df.drop(columns=['Label'])
    
    # Clean column names
    feature_df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in feature_df.columns]
    
    # Create DataFrame with exact feature structure from training
    full_feature_df = pd.DataFrame(0, index=feature_df.index, columns=feature_info['all_features'])
    
    # Fill in numerical features
    for col in feature_info['numerical_features']:
        if col in feature_df.columns:
            full_feature_df[col] = feature_df[col]
    
    # Fill in categorical features if any
    for col in feature_info['categorical_features']:
        if col in feature_df.columns:
            full_feature_df[col] = feature_df[col]
    
    # Ensure exact column order
    full_feature_df = full_feature_df[feature_info['all_features']]
    
    return full_feature_df

if __name__ == "__main__":
    # Setup GUI for file selection
    root = tk.Tk()
    root.withdraw()

    # Select model directory
    model_dir = filedialog.askdirectory(title="Select Model Directory")
    if not model_dir:
        print("No model directory selected. Exiting.")
        exit()

    # Load model components
    print("Loading model components...")
    model, dlls, entropies, feature_info = load_model(model_dir)
    print("Model features:", feature_info['names'])

    # Select test data file
    test_file = filedialog.askopenfilename(title="Select Test Data CSV", filetypes=[("MBY virus", "*.*")])
    if not test_file:
        print("No test file selected. Exiting.")
        exit()

    # Load and process test data
    print("Processing test data...")
    test_data = pd.read_csv(test_file)
    X_test = process_data(test_data, dlls, entropies, feature_info)
    
    print("Test data features order:", X_test.columns.tolist())

    # Make predictions
    print("Making predictions...")
    # Use the complete pipeline for predictions
    
    predictions = model.predict(X_test)
    
    # If test data has labels, calculate accuracy
    if 'Label' in test_data.columns:
        accuracy = accuracy_score(test_data['Label'], predictions)
        print(f"Accuracy on test data: {accuracy:.4f}")
    
    # Print predictions
    print("\nPredictions:")
    print(predictions)