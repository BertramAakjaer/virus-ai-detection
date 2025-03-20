import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

# Create a DataFrame
data = pd.DataFrame({
    'Num Feature 1': [10, 5, 15],
    'Num Feature 2': [20, 15, 25],
    'Num Feature 3': [30, 25, 35],
    'Num Feature 4': [40, 35, 45],
    'Dictionary': [
        {"WSOCK32.dll": ["bind", "listen"], "API": ["CreateProcess", "ReadFile"]},
        {"KERNEL32.dll": ["read", "write"]},
        {"USER32.dll": ["message", "input"], "API": ["SendMessage"]}
    ],
    'Label': ['Malicious', 'Benign', 'Benign']
})

# Preprocess numerical features
X_num = data[['Num Feature 1', 'Num Feature 2', 'Num Feature 3', 'Num Feature 4']]
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Preprocess dictionary features
# Flatten the dictionary values into a single list of strings
dict_values = data['Dictionary'].apply(lambda x: ' '.join(sum(x.values(), [])))

# Vectorize the dictionary values
vectorizer = CountVectorizer()
X_dict_vectorized = vectorizer.fit_transform(dict_values)

# Combine numerical and dictionary features
X_combined = hstack([X_num_scaled, X_dict_vectorized])

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