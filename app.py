from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Import numpy as needed for your model (if applicable)
# import numpy as np

from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load the pickled model
with open(os.path.join(os.path.dirname(__file__), 'svm_model_rbf.pkl'), 'rb') as f:
    model = pickle.load(f)

# Load the pre-trained TfidfVectorizer (assuming it's already fitted)
with open(os.path.join(os.path.dirname(__file__), 'tf-idf-vectorizer.pkl'), 'rb') as f:
    tfidf = pickle.load(f)

@app.route("/")
@cross_origin(origin="*",headers=['Content-Type'])
def hello():
    return "Spam Email Model API!"

@app.route('/predict', methods=['POST'])
@cross_origin(origin="*",headers=['Content-Type'])
def predict():
    data = request.json.get('data')

    if not isinstance(data, str):
        return jsonify({
            "status": "error",
            "code": 400,
            "message": "Invalid input: 'data' must be a string"
        }), 400

    try:
        # Mengubah teks menjadi representasi TF-IDF
        text_vector = tfidf.transform([data]).toarray()

        # Make the prediction
        prediction = model.predict(text_vector)
        prediction_str = str(prediction[0])

        probability = model.predict_proba(text_vector)
        real_proba = np.max(probability, axis=1)
        probability_str = str(real_proba[0])

        print(prediction)
        print(probability_str)

        return jsonify({
            "status": "success",
            "code": 200,
            "data": {
                'prediction': prediction_str,
                'probability': probability_str,
                }
            }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "code": 500,
            "message": str(e)
            }), 500

if __name__ == "__main__":
    app.run(debug=True)
