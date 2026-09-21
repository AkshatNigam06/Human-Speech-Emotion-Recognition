import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


# Load dataset
X = np.load("../dataset/X.npy")
y = np.load("../dataset/y.npy")


# Convert emotion names into numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of features:", X_train.shape[1])
# Create SVM model
model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("SVM model trained successfully!")
from sklearn.metrics import accuracy_score

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

disp.plot(xticks_rotation=45)
plt.title("Emotion Recognition Confusion Matrix")
plt.tight_layout()
plt.show()

import joblib
import os

os.makedirs("../models", exist_ok=True)

joblib.dump(model, "../models/emotion_model.pkl")
joblib.dump(encoder, "../models/label_encoder.pkl")

print("Model saved successfully!")