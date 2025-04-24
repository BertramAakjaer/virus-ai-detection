import ast, os, tkinter as tk, pandas as pd
from tkinter import filedialog

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
import numpy as np
from datetime import datetime


#TEST_SIZE = 0.01  # Proportion of data to use for testing
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

    # Updated param_grid with more strategic values
    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],  # Wider range with logarithmic scale
        'classifier__kernel': ['rbf', 'linear'],  # Added linear as it can work well for high-dimensional data
        'classifier__gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1.0],  # More granular gamma values
        'classifier__class_weight': ['balanced', None],
        'classifier__shrinking': [True]  # Simplified as this rarely impacts performance significantly
    }

    # Create GridSearchCV object
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=10
    )

    print("Starting grid search...")
    grid_search.fit(x, y)
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
            validation_fraction=0.2,  # Increased validation fraction
            solver='adam',
            early_stopping=True,
            activation='relu',
            n_iter_no_change=10  # Added patience parameter
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

    # Updated param_grid with more efficient architecture search
    param_grid = {
        'classifier__hidden_layer_sizes': [
            (64,), (128,), (256,),  # Single layer networks
            (128, 64), (256, 128), (512, 256),  # Two layer networks
            (256, 128, 64), (512, 256, 128),  # Three layer networks
            (512, 256, 128, 64)  # Deep network
        ],
        'classifier__alpha': [1e-5, 1e-4, 1e-3],  # Regularization strength
        'classifier__learning_rate_init': [0.0001, 0.001, 0.01],  # Learning rate
        'classifier__batch_size': [32, 64, 128, 256],  # Batch sizes
        'classifier__learning_rate': ['adaptive']  # Simplified to most reliable option
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

def train_model_RF(x, y, numeric_features, categorical_features):
    # Create preprocessing steps
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    # Updated param_grid with more comprehensive parameter search
    param_grid = {
        'classifier__n_estimators': [100, 200, 500],  # Increased number of trees
        'classifier__max_depth': [None, 20, 50, 100],  # Added deeper options
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None],  # Added feature selection options
        'classifier__class_weight': ['balanced', 'balanced_subsample', None],  # Added class weight options
        'classifier__bootstrap': [True]  # Simplified as False rarely improves performance
    }

    # Create GridSearchCV object
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=10
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

def save_model(model, X, model_type):
    model_type = model_type.upper()
    # Create a folder named 'trained_models' if it doesn't exist
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    models_dir = f"[{model_type}] trained_models({timestamp})"
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

    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print(f"Dataset memory usage: {data.memory_usage().sum() / 1024**2:.2f} MB")
    print(f"Number of features: {len(x.columns)}")
    print(f"Number of samples: {len(x)}")

    print("Training model... \n")
    MODEL_TYPE = input("Select model type (SVM/NN/RF): ").strip().lower()
    if MODEL_TYPE == 'svm':
        model = train_model_SVM(x, y, numeric_features, categorical_features)
    elif MODEL_TYPE == 'nn':
        model = train_model_NN(x, y, numeric_features, categorical_features)
    elif MODEL_TYPE == 'rf':
        model = train_model_RF(x, y, numeric_features, categorical_features)
    else:
        print("Invalid model type. Exiting.")
        exit()
    

    input = input("Save the model? (y/n): ")
    if input.lower() != 'y':
        print("Model not saved.")
        exit()

    save_model(model, x, MODEL_TYPE)