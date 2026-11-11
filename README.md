# 🎙️ Human Emotion Recognition Through Speech

An AI-powered Speech Emotion Recognition (SER) web application that analyzes human speech and predicts the emotion expressed through voice.

🌐 **Live Demo:** https://human-speech-emotion-recognition.onrender.com/

💻 **GitHub Repository:** https://github.com/AkshatNigam06/Human-Speech-Emotion-Recognition

## 📌 Project Overview

Human Emotion Recognition Through Speech is a machine learning project designed to identify emotions from audio signals. It extracts meaningful acoustic features from speech and uses trained machine learning models to classify the speaker's emotional state.

The application provides an interactive web interface where users can upload an audio file or record their voice directly to get an emotion prediction.

## ✨ Features

- 🎤 **Live Voice Recording:** Record speech directly through the browser.
- 📁 **Audio Upload:** Upload audio files for emotion analysis.
- 🧠 **Machine Learning Prediction:** Predict emotions using trained ML models.
- 🎵 **MFCC Feature Extraction:** Extract acoustic features from speech signals.
- 📊 **Multiple Models:** Experiment with SVM, Random Forest, and KNN.
- 🌐 **Web Application:** User-friendly frontend integrated with a Flask backend.
- 🚀 **Live Deployment:** Hosted on Render.

## 😊 Emotions Recognized

The model is trained to recognize eight emotion categories:

1. Neutral
2. Calm
3. Happy
4. Sad
5. Angry
6. Fearful
7. Disgust
8. Surprised

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Frontend | HTML, CSS, JavaScript |
| Backend | Flask |
| Machine Learning | Scikit-learn |
| Audio Processing | Librosa, SoundFile |
| Feature Extraction | MFCC |
| Data Processing | NumPy, Pandas |
| Model Storage | Joblib / Pickle |
| Deployment | Render |
| Version Control | Git & GitHub |

## 🧠 Machine Learning Models

The project includes experiments with the following algorithms:

- Support Vector Machine (SVM)
- Random Forest Classifier
- K-Nearest Neighbors (KNN)

The application uses the saved trained model for making predictions.

## 📂 Project Structure

```text
Human-Speech-Emotion-Recognition/
│
├── backend/
│   ├── app.py
│   ├── feature_extraction.py
│   ├── model.py
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── dataset/
│   └── Processed Dataset Files
│
├── models/
│   ├── emotion_model.pkl
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   ├── knn_model.pkl
│   └── label_encoder.pkl
│
├── graphs/
│
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AkshatNigam06/Human-Speech-Emotion-Recognition.git
```

### 2. Navigate to the Project Directory

```bash
cd Human-Speech-Emotion-Recognition
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 6. Run the Flask Application

```bash
cd backend
python app.py
```

### 7. Open in Browser

```text
http://127.0.0.1:5000
```

## 🚀 Live Application

Try the deployed application:

**https://human-speech-emotion-recognition.onrender.com/**

Upload an audio file or record your voice to get an emotion prediction.

## 📚 Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset for speech emotion recognition.

The dataset contains audio recordings representing different emotional expressions. Processed features are used for model training and evaluation.

Original dataset audio files are not included in this repository.

## 🔮 Future Improvements

- Improve model accuracy and generalization.
- Explore deep learning architectures such as CNN and BiLSTM.
- Improve real-time emotion prediction.
- Add detailed model performance visualizations.
- Enhance the user interface and prediction reports.

## 👨‍💻 Author

**Akshat Nigam**

- GitHub: https://github.com/AkshatNigam06
- Portfolio: https://akshatnigam06.github.io/portfolio/

---

⭐ If you find this project interesting, consider giving the repository a star!

**Built with Python, Machine Learning, and Flask.**
