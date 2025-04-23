import ast
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime

def setup_data(data):
    data = data.dropna()
    
    # Convert string features to dictionaries
    data['EntropyAndSections'] = data['EntropyAndSections'].apply(ast.literal_eval)
    data['ImportedDLLS'] = data['ImportedDLLS'].apply(ast.literal_eval)

    # Extract DLL features
    all_dlls = set()
    for dll_dict in data['ImportedDLLS']:
        all_dlls.update(dll_dict.keys())
    dll_counts = {f'dll_{dll}_count': data['ImportedDLLS'].apply(lambda x: len(x.get(dll, []))) 
                 for dll in all_dlls}
    dll_df = pd.DataFrame(dll_counts)

    # Extract entropy features
    all_entropies = set()
    for entropy_dict in data['EntropyAndSections']:
        all_entropies.update(entropy_dict.keys())
    entropy_counts = {f'entropy_{entropy}_count': data['EntropyAndSections'].apply(lambda x: len(str(x.get(entropy, []))))
                     for entropy in all_entropies}
    entropy_df = pd.DataFrame(entropy_counts)

    # Combine features
    data = pd.concat([data, dll_df, entropy_df], axis=1)
    X = data.drop(columns=['EntropyAndSections', 'ImportedDLLS', 'Label', 'Name'])
    X.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X.columns]
    
    # Check and remove duplicate columns
    duplicate_cols = X.columns[X.columns.duplicated()]
    if len(duplicate_cols) > 0:
        print(f"Found duplicate columns: {duplicate_cols}")
        print("Removing duplicates...")
        X = X.loc[:, ~X.columns.duplicated()]
    
    y = data['Label']

    # Setup preprocessor
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    preprocessor = ColumnTransformer([('num', StandardScaler(), numerical_features)])
    
    return X, y, preprocessor, all_dlls, all_entropies


def export_training_data(X, y):
    """Export training data to Excel with features and labels"""
    training_data = pd.concat([X, y], axis=1)
    excel_path = "training_data_features.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        training_data.to_excel(writer, sheet_name='Training Data', index=False)
    print(f"Training data exported to {excel_path}")

def save_model(model, dll, entropy, X):
    models_dir = f"trained_models({datetime.now().strftime('%Y-%m-%d %H-%M-%S')})"
    os.makedirs(models_dir, exist_ok=True)

    # Save model and features
    joblib.dump(model, os.path.join(models_dir, "model_pipeline.pkl"))
    joblib.dump(dll, os.path.join(models_dir, "dlls.pkl"))
    joblib.dump(entropy, os.path.join(models_dir, "entropies.pkl"))
    
    # Save feature names
    num_features = model.named_steps['preprocessor'].transformers_[0][2]
    feature_info = {
        'numerical_features': list(num_features),
        'categorical_features': [],
        'all_features': list(num_features)
    }
    joblib.dump(feature_info, os.path.join(models_dir, "feature_names.pkl"))
    
    
    """Export an empty CSV with column headers for new data"""
    # Create empty DataFrame with same columns
    template = pd.DataFrame(columns = X.columns.tolist())
    template_path = os.path.join(models_dir, "template.csv")
    template.to_csv(template_path, index=False)
    print(f"Empty template exported to {template_path}")
    
    
    print(f"Model saved in '{models_dir}'")

if __name__ == "__main__":
    # Load data
    data = pd.read_csv("[2] combined_data.csv")
    print("Data loaded, shape:", data.shape)

    # Setup and split data
    X, y, preprocessor, dlls, entropies = setup_data(data)
    
    # Export training data and template
    export_training_data(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    print(f"Training accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Testing accuracy: {model.score(X_test, y_test):.4f}")
    
    
    print("dlls:", dlls)
    print("entropies:", entropies)

    # Save model
    save_model(model, dlls, entropies, X)
