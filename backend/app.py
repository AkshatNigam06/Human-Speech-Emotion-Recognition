from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
from feature_extraction import extract_mfcc

app = Flask(__name__)

# Load trained model
model = joblib.load("../models/emotion_model.pkl")
encoder = joblib.load("../models/label_encoder.pkl")


@app.route("/")
def home():
    return send_from_directory("../frontend", "index.html")
@app.route("/style.css")
def style():
    return send_from_directory("../frontend", "style.css")


@app.route("/script.js")
def script():
    return send_from_directory("../frontend", "script.js")


@app.route("/predict", methods=["POST"])
def predict():

    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]

    temp_path = "../dataset/temp_audio.wav"
    audio_file.save(temp_path)

    try:
        # Extract MFCC features
        features = extract_mfcc(temp_path)

        # Reshape for model
        features = features.reshape(1, -1)

        # Prediction
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)[0]

        emotion = encoder.inverse_transform(prediction)[0]
        confidence = max(probabilities) * 100

        probability_dict = {
            encoder.classes_[i]: round(probabilities[i] * 100, 2)
            for i in range(len(encoder.classes_))
        }

        return jsonify({
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "probabilities": probability_dict
        })

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(debug=True)