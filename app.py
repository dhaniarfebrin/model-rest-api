from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import string
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load the pickled model
with open('svm_model_rbf.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

@app.route("/")
@cross_origin(origin="*",headers=['Content-Type'])
def hello():
    return "Spam Email Model API!"

@app.route('/predict', methods=['POST'])
@cross_origin(origin="*",headers=['Content-Type'])
def predict():
    data = request.json['data']

    # Mengubah teks menjadi representasi TF-IDF
    text_vector = tfidf.transform([data]).toarray()

    # Make the prediction
    prediction = model.predict(text_vector)
    prediction_str = str(prediction[0])

    return jsonify({'prediction': prediction_str})

if __name__ == "__main__":
    app.run()