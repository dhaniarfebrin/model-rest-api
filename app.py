from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import PorterStemmer
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Membuat stemmer bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Download resources untuk nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    # Hapus tanda baca dari teks
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Ubah teks menjadi huruf kecil (lowercase)
    text = text.lower()

    # Tokenize teks dan hapus stop words
    # Menggunakan list comprehension untuk mengambil kata yang bukan stop words
    tokens = [word for word in text.split() if word not in stop_words]

    # Lakukan stemming pada setiap token yang tersisa menggunakan Sastrawi
    tokens = [stemmer.stem(word) for word in tokens]

    # Gabungkan kembali token yang telah di-stem menjadi satu string
    return ' '.join(tokens)

# Load the pickled model
with open('model_svm.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

@app.route("/")
@cross_origin()
def hello():
    return "Hello World!"

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.json['data']
    process_text = preprocess_text(data)

    # Mengubah teks menjadi representasi TF-IDF
    text_vector = tfidf.transform([process_text]).toarray()

    # Make the prediction
    prediction = model.predict(text_vector)
    prediction_str = str(prediction[0])

    return jsonify({'prediction': prediction_str})

if __name__ == "__main__":
    app.run(debug = True)