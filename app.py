from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the pre-trained model and TF-IDF vectorizer
model = joblib.load('spam_classifier_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    vectorized_message = tfidf_vectorizer.transform([message])
    prediction = model.predict(vectorized_message)
    is_spam = int(prediction[0])
    return render_template('index.html', message=message, is_spam=is_spam)

if __name__ == '__main__':
    app.run(debug=True)