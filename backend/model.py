
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
GRAPH_DIR = os.path.join(PROJECT_DIR, "graphs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

X = np.load(os.path.join(DATASET_DIR, "X.npy"))
y = np.load(os.path.join(DATASET_DIR, "y.npy"))
actors = np.load(os.path.join(DATASET_DIR, "actors.npy"))

print("\nDataset Loaded Successfully!")
print("Features Shape:", X.shape)
print("Labels Shape:", y.shape)
print("Actors Shape:", actors.shape)

# Validate dataset
if not (len(X) == len(y) == len(actors)):
    raise ValueError("X, y and actors must have equal lengths!")

if X.ndim != 2:
    raise ValueError("X must be a 2D feature array!")

if len(np.unique(actors)) < 2:
    raise ValueError("At least two actors are required for splitting!")


# ==========================================
# 3. ENCODE EMOTION LABELS
# ==========================================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nEmotion Classes:")
print(encoder.classes_)


# ==========================================
# 4. ACTOR-INDEPENDENT TRAIN-TEST SPLIT
# ==========================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y_encoded, groups=actors)
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y_encoded[train_idx]
y_test = y_encoded[test_idx]

train_actors = np.unique(actors[train_idx])
test_actors = np.unique(actors[test_idx])

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("Training Actors:", train_actors)
print("Testing Actors:", test_actors)

# Confirm no actor leakage
if set(train_actors).intersection(set(test_actors)):
    raise ValueError("Actor leakage detected!")

print("\nActor-independent split verified!")


# ==========================================
# 5. DEFINE MACHINE LEARNING MODELS
# ==========================================

models = {

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        )
    )
}


# ==========================================
# 6. TRAIN AND EVALUATE MODELS
# ==========================================

results = []
trained_models = {}

for model_name, model in models.items():

    print("\n" + "=" * 55)
    print("Training:", model_name)
    print("=" * 55)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\nAccuracy:", round(accuracy * 100, 2), "%")
    print("Macro Precision:", round(precision * 100, 2), "%")
    print("Macro Recall:", round(recall * 100, 2), "%")
    print("Macro F1-Score:", round(f1 * 100, 2), "%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0
        )
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1-Score": f1 * 100
    })

    trained_models[model_name] = model

    # Save individual model
    model_filename = (
        model_name.lower().replace(" ", "_") + "_model.pkl"
    )

    joblib.dump(
        model,
        os.path.join(MODEL_DIR, model_filename)
    )

    # Generate confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(encoder.classes_))
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=encoder.classes_
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False
    )

    ax.set_title(f"{model_name} Confusion Matrix")
    fig.tight_layout()

    cm_filename = (
        model_name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    fig.savefig(
        os.path.join(GRAPH_DIR, cm_filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==========================================
# 7. MODEL PERFORMANCE COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)

print(results_df.round(2).to_string(index=False))

# Save comparison CSV
results_df.to_csv(
    os.path.join(GRAPH_DIR, "model_comparison.csv"),
    index=False
)


# ==========================================
# 8. SELECT AND SAVE BEST MODEL
# ==========================================

best_model_name = results_df.loc[
    results_df["F1-Score"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)

joblib.dump(
    best_model,
    os.path.join(MODEL_DIR, "emotion_model.pkl")
)

joblib.dump(
    encoder,
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)

# Save split information for reproducibility
split_info = {
    "training_actors": train_actors.tolist(),
    "testing_actors": test_actors.tolist(),
    "best_model": best_model_name
}

joblib.dump(
    split_info,
    os.path.join(MODEL_DIR, "split_info.pkl")
)


# ==========================================
# 9. MODEL COMPARISON GRAPH
# ==========================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

ax = results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6),
    width=0.75
)

ax.set_title(
    "Speech Emotion Recognition - Model Comparison"
)

ax.set_ylabel("Score (%)")
ax.set_xlabel("Machine Learning Models")
ax.set_ylim(0, 100)

ax.legend(loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "model_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 10. FINAL OUTPUT
# ==========================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("Best Model:", best_model_name)

print(
    "Saved Model:",
    os.path.join(MODEL_DIR, "emotion_model.pkl")
)

print(
    "Saved Encoder:",
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)

print(
    "Comparison CSV:",
    os.path.join(GRAPH_DIR, "model_comparison.csv")
)

print(
    "Comparison Graph:",
    os.path.join(GRAPH_DIR, "model_comparison.png")
)

print("\nTraining and testing actors are separate.")
