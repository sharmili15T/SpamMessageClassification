import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
data = pd.read_csv('spam_dataset.csv')

# Preprocess the text
data['message'] = data['message'].apply(lambda x: x.lower())  # Convert to lowercase
# ... Additional text preprocessing steps

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Convert text data into numerical vectors
tfidf_vectorizer = TfidfVectorizer()
X_train_vectors = tfidf_vectorizer.fit_transform(X_train)
X_test_vectors = tfidf_vectorizer.transform(X_test)

# Code for model training
from sklearn.svm import SVC
# Create and train the model
model = SVC(kernel='linear')
model.fit(X_train_vectors, y_train)

# Model evaluation
from sklearn.metrics import accuracy_score, classification_report
# Predict on the test set
y_pred = model.predict(X_test_vectors)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

