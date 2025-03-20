import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

# Create a DataFrame
data = pd.DataFrame({
    'Numerical Feature': [10, 5, 15],
    'Dictionaries': [
        [{"WSOCK32.dll": ["bind", "listen"]}, {"API": ["CreateProcess", "ReadFile"]}],
        [{"KERNEL32.dll": ["read", "write"]}],
        [{"USER32.dll": ["message", "input"]}, {"API": ["SendMessage"]}]
    ],
    'Label': ['Malicious', 'Benign', 'Benign']
})

# Preprocess numerical features
X_num = data[['Numerical Feature']]
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Preprocess dictionaries
vectorizers = [CountVectorizer() for _ in range(data['Dictionaries'].apply(len).max())]

# Vectorize each dictionary separately
X_dict_vectorized_list = []
for i in range(len(vectorizers)):
    dict_values = data['Dictionaries'].apply(
        lambda x: ' '.join(sum(x[i].values(), [])) if i < len(x) else ''
    )
    X_dict_vectorized = vectorizers[i].fit_transform(dict_values)
    X_dict_vectorized_list.append(X_dict_vectorized)

# Combine numerical and dictionary features
X_combined = hstack([X_num_scaled] + X_dict_vectorized_list)

# Target variable
y = data['Label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
