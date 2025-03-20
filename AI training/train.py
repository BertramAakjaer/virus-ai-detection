import ast, os, joblib, tkinter as tk, pandas as pd
from tkinter import filedialog
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from scipy.sparse import hstack
from tqdm import tqdm





# Training configuration
EPOCHS = 10  # Number of training iterations
TEST_SIZE = 0.2  # Proportion of data to use for testing
RANDOM_STATE = 42  # Seed for reproducibility
CLASS_WEIGHT = 'balanced'  # Options: None, 'balanced', or a dictionary {class_label: weight}
SCALER_TYPE = 'standard'  # Options: 'standard', 'minmax'

# Hyperparameter tuning
LOGISTIC_PARAMS = {'C': 1.0, 'solver': 'lbfgs'}  # Logistic Regression parameters
GRID_SEARCH = False  # Enable or disable GridSearchCV
GRID_PARAM_GRID = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}  # GridSearchCV parameters

# Feature engineering
USE_TFIDF = True  # Use TF-IDF instead of CountVectorizer
N_GRAMS = (1, 2)  # N-grams for vectorization

# Evaluation metrics
CALCULATE_ROC_AUC = True  # Whether to calculate ROC-AUC




def setup_data(data):
    # Preprocess numerical features from CSV
    X_num = data[['SizeOfCode', 'SizeOfInitializedData', 'SizeOfImage', 'Subsystem']]
    
    # Use the scaler type
    if SCALER_TYPE == 'standard':
        scaler = StandardScaler()
    elif SCALER_TYPE == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    X_num_scaled = scaler.fit_transform(X_num)

    # Parse and preprocess entropy and sections features
    data['EntropyAndSections'] = data['EntropyAndSections'].apply(ast.literal_eval)
    entropy_df = pd.json_normalize(data['EntropyAndSections'])
    entropy_df.fillna(0, inplace=True)
    entropy_scaler = StandardScaler()
    entropy_scaled = entropy_scaler.fit_transform(entropy_df)

    # Parse and preprocess dictionary features from 'ImportetDLLS'
    data['ImportedDLLS'] = data['ImportedDLLS'].apply(ast.literal_eval)
    dict_values = data['ImportedDLLS'].apply(lambda x: ' '.join(sum(x.values(), [])))
    
    # Use TF-IDF if enabled
    if USE_TFIDF:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(ngram_range=N_GRAMS)
    else:
        vectorizer = CountVectorizer()
    
    X_dict_vectorized = vectorizer.fit_transform(dict_values)

    # Combine numerical, entropy, and dictionary features
    X_combined = hstack([X_num_scaled, entropy_scaled, X_dict_vectorized])

    # Target variable
    y = data['Label']

    preprocessing_components = {
        'scaler': scaler,
        'entropy_scaler': entropy_scaler,
        'vectorizer': vectorizer
    }

    return X_combined, y, preprocessing_components


def train_model(x_train, y_train):
    if GRID_SEARCH:
        grid_search = GridSearchCV(LogisticRegression(), GRID_PARAM_GRID, cv=5)
        grid_search.fit(x_train, y_train)
        model = grid_search.best_estimator_
    else:
        model = LogisticRegression(**LOGISTIC_PARAMS, class_weight=CLASS_WEIGHT)

    for _ in tqdm(range(EPOCHS), desc="Training Progress"):
        model.fit(x_train, y_train)
    
    return model


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    scores = cross_val_score(model, x, y, cv=5)
    print(f"Cross-validation accuracy: {scores.mean()}")

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])}")

def save_model(model, preprocessing):
    # Create a folder named 'trained_models' if it doesn't exist
    models_dir = "trained_models"
    os.makedirs(models_dir, exist_ok=True)

    # Save the model
    main_model_path = os.path.join(models_dir, "logistic_regression_model.pkl")
    preprocessing_path = os.path.join(models_dir, "preprocessing_components.pkl")

    joblib.dump(model, main_model_path)
    joblib.dump(preprocessing, preprocessing_path)

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
    x, y, preprocessing = setup_data(data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print("Training model... \n")
    model = train_model()
    

    print("Model trained. Evaluating...")
    evaluate_model(model, x_test, y_train, y_test)





    input = input("Save the model? (y/n): ")
    if input.lower() != 'y':
        print("Model not saved.")
        exit()

    save_model(model, preprocessing)