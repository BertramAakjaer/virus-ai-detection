import pandas as pd
import ast
from scipy.sparse import hstack
import joblib

def preprocess_data(data, preprocessing):
    """Process new data using saved preprocessing components"""
    # Extract preprocessing components
    scaler = preprocessing['scaler']
    entropy_scaler = preprocessing['entropy_scaler']
    vectorizer = preprocessing['vectorizer']
    
    # Preprocess numerical features
    X_num = data[['SizeOfCode', 'SizeOfInitializedData', 'SizeOfImage', 'Subsystem']]
    X_num_scaled = scaler.transform(X_num)
    
    # Preprocess entropy and sections
    data['EntropyAndSections'] = data['EntropyAndSections'].apply(ast.literal_eval)
    entropy_df = pd.json_normalize(data['EntropyAndSections'])
    entropy_df.fillna(0, inplace=True)
    # Ensure all columns from training are present (add with zeros if missing)
    missing_cols = set(entropy_scaler.feature_names_in_) - set(entropy_df.columns)
    for col in missing_cols:
        entropy_df[col] = 0
    entropy_df = entropy_df[entropy_scaler.feature_names_in_]  # Reorder columns
    entropy_scaled = entropy_scaler.transform(entropy_df)
    
    # Preprocess dictionary features
    data['ImportedDLLS'] = data['ImportedDLLS'].apply(ast.literal_eval)
    dict_values = data['ImportedDLLS'].apply(lambda x: ' '.join(sum(x.values(), [])))
    X_dict_vectorized = vectorizer.transform(dict_values)
    
    # Combine features
    X_combined = hstack([X_num_scaled, entropy_scaled, X_dict_vectorized])
    
    return X_combined

# Load the model and preprocessing components
model = joblib.load('virus_detection_model.pkl')
preprocessing = joblib.load('preprocessing_components.pkl')

# Load new data for prediction
new_data = pd.read_csv('new_samples.csv')  # Replace with your data file

# Preprocess the data
X_new = preprocess_data(new_data, preprocessing)

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)

# Print or save results
print("Predictions:", predictions)
print("Probability of malware:", probabilities[:, 1])  # Assuming class 1 is malware

# Add predictions to the dataframe
new_data['Prediction'] = predictions
new_data.to_csv('prediction_results.csv', index=False)