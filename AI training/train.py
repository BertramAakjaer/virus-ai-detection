import ast, os, tkinter as tk, pandas as pd
from tkinter import filedialog

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
import numpy as np
from datetime import datetime


TEST_SIZE = 0.01  # Proportion of data to use for testing
RANDOM_STATE = 60  # Seed for reproducibility


# Moving the data into correct format for training

def setup_data(data):
    # Create an explicit copy of the data
    data = data.copy()
    data = data.dropna()

    # Convert string features to dictionaries using loc
    data.loc[:, 'EntropyAndSections'] = data['EntropyAndSections'].apply(ast.literal_eval)
    data.loc[:, 'ImportedDLLS'] = data['ImportedDLLS'].apply(ast.literal_eval)

    print("\nDataset shape:", data.shape)

    # Define numeric and categorical features
    numeric_features = ['SizeOfCode', 'SizeOfInitializedData', 'SizeOfImage']
    categorical_features = ['Subsystem']

    # Process DLLs and Entropy sections as before
    all_dlls = set()
    for dll_dict in data['ImportedDLLS']:
        all_dlls.update(dll_dict.keys())

    dll_counts = {f'dll_{dll}_count': data['ImportedDLLS'].apply(lambda x: len(x.get(dll, []))) 
                 for dll in all_dlls}
    dll_df = pd.DataFrame(dll_counts)

    all_entropies = set()
    for entropy_dict in data['EntropyAndSections']:
        all_entropies.update(entropy_dict.keys())

    entropy_counts = {f'entropy_{entropy}_count': data['EntropyAndSections'].apply(lambda x: len(str(x.get(entropy, ''))))
                     for entropy in all_entropies}
    entropy_df = pd.DataFrame(entropy_counts)

    # Combine all features
    data = pd.concat([data, dll_df, entropy_df], axis=1)

    # Add DLL and entropy columns to numeric features
    numeric_features.extend(dll_df.columns.tolist())
    numeric_features.extend(entropy_df.columns.tolist())

    # Prepare input features and target variable
    X = data[numeric_features + categorical_features]
    
    # Convert text labels to binary values (0 for harmless, 1 for malware)
    y = (data['Label'] == 'malware').astype(int)
    
    print("\nLabel distribution:")
    print(pd.Series(y).value_counts().to_frame(name='count'))
    print("\nLabel mapping:")
    print("0 = harmless")
    print("1 = malware")

    # Clean column names
    cleaned_columns = []
    for col in X.columns:
        cleaned_name = ""
        for char in str(col):
            if char.isalnum():
                cleaned_name += char
            else:
                cleaned_name += "_"
        cleaned_columns.append(cleaned_name)
    
    X.columns = cleaned_columns
    
    # Update feature lists with cleaned names
    numeric_features = [col for col in X.columns if col not in ['Subsystem']]
    categorical_features = ['Subsystem']

    # Remove any duplicate columns
    X = X.loc[:, ~X.columns.duplicated()]
    
    return X, y, numeric_features, categorical_features


# Training of the model

def train_model_SVM(x, y, numeric_features, categorical_features):
    # Create preprocessing steps
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the pipeline with probability=True
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(random_state=RANDOM_STATE, probability=True))
    ])

    # Simplify param_grid to remove probability parameter since it must be True
    param_grid = {
        'classifier__C': [0.01, 0.03, 0.09, 0.27, 1, 3, 9, 27, 81, 243, 729, 2187, 6561],
        'classifier__kernel': ['rbf'], # 'linear', 'poly', 'sigmoid'],
        'classifier__degree': [1, 2, 3, 4, 6],
        'classifier__shrinking': [True, False],
        'classifier__class_weight': [None, 'balanced'],
        'classifier__gamma': ['scale', 'auto', 0.1]
    }

    # Create GridSearchCV object
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    # Fit the grid search
    print("Starting grid search...")
    grid_search.fit(x, y)

    # Print best parameters
    print("\nBest parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    return grid_search.best_estimator_

def train_model_NN(x, y, numeric_features, categorical_features):
    # Create preprocessing steps
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(sparse_output=False)

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Define the pipeline with binary classification specifics
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(
            random_state=RANDOM_STATE,
            max_iter=10000,
            validation_fraction=0.1,
            solver='adam',  # Better for binary classification
            early_stopping=True,
            activation='relu',
        ))
    ])

    # Ensure binary classification labels
    x = x.astype(np.float32)
    y = y.astype(np.int32)
    
    # Verify unique classes
    unique_classes = np.unique(y)
    print(f"Unique classes in dataset: {unique_classes}")
    
    if len(unique_classes) != 2:
        raise ValueError(f"Expected binary classification (2 classes), but got {len(unique_classes)} classes")

    param_grid = {
        'classifier__hidden_layer_sizes': [
            (8,), (16,), (32,), (64,), (128,), (256,), 
            (32, 16), (64, 32), (128, 64), (256, 128),
            (32, 16, 8), (64, 32, 16), (128, 64, 32), (256, 128, 64),
            (32, 16, 8, 4), (64, 32, 16, 8)

        ],
        'classifier__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'classifier__learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
        'classifier__batch_size': [4, 8, 16, 32, 64, 128, 256, 500],
        'classifier__learning_rate': ['constant', 'adaptive', 'invscaling'],
    }

    # Create GridSearchCV object
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=10,
        error_score='raise'
    )

    print("Starting grid search...")
    grid_search.fit(x, y)
    print("\nBest parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    return grid_search.best_estimator_
    
    

# Evaultation of model

def predict(model, x):
    # Make predictions on new data
    predictions = model.predict(x)
    return predictions


# Saving model for later use

def save_model(model, X):
    # Create a folder named 'trained_models' if it doesn't exist
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    models_dir = f"trained_models({timestamp})"
    os.makedirs(models_dir, exist_ok=True)

    # Save the complete model pipeline and additional data
    model_path = os.path.join(models_dir, "model_pipeline.pkl")

    joblib.dump(model, model_path)
    
    """Export an empty CSV with column headers for new data"""
    # Create empty DataFrame with same columns
    template = pd.DataFrame(columns = X.columns.tolist())
    template_path = os.path.join(models_dir, "template.csv")
    template.to_csv(template_path, index=False)
    print(f"Empty template exported to {template_path}")

    print(f"Model and preprocessing tools saved in '{models_dir}'")




###########################################################
##                  Main Function                        ##
###########################################################


if __name__ == "__main__":
    
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        print("No file selected. Exiting.")
        exit()

    print(f"Selected file: {file_path}")
    print("Loading data...")
    data = pd.read_csv(file_path)
    
    # Add this after loading data

    print("Setting up data for training...")
    x, y, numeric_features, categorical_features = setup_data(data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print(f"Dataset memory usage: {data.memory_usage().sum() / 1024**2:.2f} MB")
    print(f"Number of features: {len(x.columns)}")
    print(f"Number of samples: {len(x)}")

    print("Training model... \n")
    model = train_model_NN(x_train, y_train, numeric_features, categorical_features)
    

    print("Model trained. Evaluating...")
    predictions = predict(model, x_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    

    input = input("Save the model? (y/n): ")
    if input.lower() != 'y':
        print("Model not saved.")
        exit()

    save_model(model, x)