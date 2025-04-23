import ast, os, tkinter as tk, pandas as pd
from tkinter import filedialog


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
from datetime import datetime


TEST_SIZE = 0.3  # Proportion of data to use for testing
RANDOM_STATE = 10  # Seed for reproducibility


def setup_data(data):
    data = data.dropna()

    data['EntropyAndSections'] = data['EntropyAndSections'].apply(ast.literal_eval)
    data['ImportedDLLS'] = data['ImportedDLLS'].apply(ast.literal_eval)

    #print("\nColumns in dataset:")
    #for col in data.columns:
    #    print(col)
        
    print("\nDataset shape:", data.shape)

    feature_columns = ['SizeOfCode', 'SizeOfInitializedData', 'SizeOfImage', 'Subsystem', 'EntropyAndSections', 'ImportedDLLS']
    target_column = 'Label'

    X = data[feature_columns]
    y = data[target_column]

    print("Features (X) head:\n", X.tail())
    print("\nTarget (y) head:\n", y.tail())
    print("\nTarget value counts:\n", y.value_counts()) # Good to check class distribution

    all_dlls = set()
    for dll_dict in data['ImportedDLLS']:
        all_dlls.update(dll_dict.keys())

    dll_counts = {}
    for dll in all_dlls:
        dll_counts[f'dll_{dll}_count'] = data['ImportedDLLS'].apply(lambda x: len(x.get(dll, [])))
    dll_df = pd.DataFrame(dll_counts)

    all_entropies = set()
    for entropy_dict in data['EntropyAndSections']:
        all_entropies.update(entropy_dict.keys())

    entropy_counts = {}
    for entropy in all_entropies:
        entropy_counts[f'entropy_{entropy}_count'] = data['EntropyAndSections'].apply(lambda x: len(str(x.get(entropy, []))))
    entropy_df = pd.DataFrame(entropy_counts)

    # Combine all features at once
    data = pd.concat([data, dll_df, entropy_df], axis=1)

    # Prepare input features and target variable
    X = data.drop(columns=['EntropyAndSections', 'ImportedDLLS', 'Label', 'Name'])
    X.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X.columns]

    y = data['Label']

    # Update numerical and categorical features after dropping columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns


    with pd.ExcelWriter('features.xlsx') as writer:
        pd.concat([X, y], axis=1).to_excel(writer, sheet_name='data')

    # Ensure column names are unique
    X = X.loc[:, ~X.columns.duplicated()]
    
    # Preprocessing steps
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return X, y, preprocessor

def train_model(x, y, preprocessor):
    # Define model
    model = RandomForestClassifier()
    # Create preprocessing and modeling pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)])

    # Model training with GridSearchCV for hyperparameter tuning
    param_grid = { 'model__n_estimators': [100, 2000, 200, 500, 300, 5000],
                    'model__max_features': ['sqrt', 'log2'],}

    # Calculate total fits that will be performed
    n_iter = len(param_grid['model__n_estimators']) * \
            len(param_grid['model__max_features']) * + 5  # Times 5 for CV folds
            
    print(f"Starting grid search with {n_iter} total fits...")


    # Create GridSearchCV with verbose output
    CV = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)

    # Fit the model - verbose=2 will show real-time progress
    CV.fit(x, y)

    # Best model parameters
    print('\nBest parameters:', CV.best_params_)
    return CV.best_estimator_

def predict(model, x):
    # Make predictions on new data
    predictions = model.predict(x)
    return predictions

def save_model(model, preprocessor, X):
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

    print("Setting up data for training...")
    x, y, preprocessing= setup_data(data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print("Training model... \n")
    model = train_model(x_train, y_train, preprocessing)
    

    print("Model trained. Evaluating...")
    predictions = predict(model, x_test)
    print("Accuracy:", accuracy_score(y_test, predictions))

    input = input("Save the model? (y/n): ")
    if input.lower() != 'y':
        print("Model not saved.")
        exit()

    save_model(model, preprocessing, x)