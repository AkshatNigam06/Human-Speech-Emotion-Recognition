
from flask import Flask, request, jsonify, send_from_directory, send_file
import joblib
import os
import tempfile
import subprocess
import imageio_ffmpeg
import io

from pathlib import Path
from feature_extraction import extract_mfcc


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL_DIR = PROJECT_DIR / "models"
FRONTEND_DIR = PROJECT_DIR / "frontend"

MODEL_PATH = MODEL_DIR / "emotion_model.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"


# ==========================================
# 2. FLASK CONFIGURATION
# ==========================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


# ==========================================
# 3. LOAD MODEL
# ==========================================

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

print("Model loaded successfully!")
print("Expected feature count:", model.n_features_in_)
print("Emotion classes:", encoder.classes_)


# ==========================================
# 4. FRONTEND ROUTES
# ==========================================

@app.route("/")
def home():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/style.css")
def style():
    return send_from_directory(str(FRONTEND_DIR), "style.css")


@app.route("/script.js")
def script():
    return send_from_directory(str(FRONTEND_DIR), "script.js")


# ==========================================
# 5. HEALTH CHECK
# ==========================================

@app.route("/health")
def health():
    return jsonify({
        "status": "success",
        "model": "Random Forest / Best Model",
        "emotions": encoder.classes_.tolist()
    })


# ==========================================
# 6. AUDIO CONVERSION
# ==========================================

def convert_to_wav(input_path, output_path):

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
        "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "22050",
        "-c:a", "pcm_s16le",
        output_path
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise ValueError(
            "Audio conversion failed: " + result.stderr[-1000:]
        )


# ==========================================
# 7. AUDIO PREVIEW API
# ==========================================

@app.route("/preview", methods=["POST"])
def preview():

    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]

    if audio_file.filename == "":
        return jsonify({"error": "No audio file selected"}), 400

    input_path = None
    output_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".webm",
            delete=False
        ) as input_file:

            input_path = input_file.name
            audio_file.save(input_path)

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as output_file:

            output_path = output_file.name

        convert_to_wav(input_path, output_path)

        with open(output_path, "rb") as wav_file:
            wav_data = wav_file.read()

        return send_file(
            io.BytesIO(wav_data),
            mimetype="audio/wav",
            as_attachment=False,
            download_name="recording.wav"
        )

    except Exception as error:

        app.logger.exception("Preview conversion failed")

        return jsonify({
            "error": str(error)
        }), 500

    finally:

        for path in [input_path, output_path]:

            if path and os.path.exists(path):
                os.remove(path)


# ==========================================
# 8. EMOTION PREDICTION API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]

    if audio_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    input_path = None
    output_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".audio",
            delete=False
        ) as input_file:

            input_path = input_file.name
            audio_file.save(input_path)

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as output_file:

            output_path = output_file.name

        convert_to_wav(input_path, output_path)

        features = extract_mfcc(output_path)

        if features.shape[0] != model.n_features_in_:
            raise ValueError(
                f"Invalid audio feature dimensions: "
                f"Expected {model.n_features_in_}, "
                f"Got {features.shape[0]}"
            )

        features = features.reshape(1, -1)

        prediction = model.predict(features)
        probabilities = model.predict_proba(features)[0]

        emotion = encoder.inverse_transform(prediction)[0]
        confidence = float(max(probabilities) * 100)

        probability_dict = {
            str(encoder.classes_[i]): round(
                float(probabilities[i]) * 100, 2
            )
            for i in range(len(encoder.classes_))
        }

        return jsonify({
            "emotion": str(emotion),
            "confidence": round(confidence, 2),
            "probabilities": probability_dict
        })

    except Exception as error:

        app.logger.exception("Prediction failed")

        return jsonify({
            "error": str(error)
        }), 500

    finally:

        for path in [input_path, output_path]:

            if path and os.path.exists(path):
                os.remove(path)


# ==========================================
# 9. ERROR HANDLER
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": "Audio file is too large. Maximum size is 20 MB."
    }), 413


# ==========================================
# 10. RUN SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
