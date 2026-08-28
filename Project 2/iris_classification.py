"""
Project 2: Data Classification Using AI
DecodeLabs Industrial Training Kit — Batch 2026

Goal
----
Build a basic classification model using a small dataset (Iris) that follows
the IPO framework from the training slides:

    INPUT   -> Iris dataset, feature scaling
    PROCESS -> Train/test split, K-Nearest Neighbors (KNN) algorithm
    OUTPUT  -> Confusion matrix, F1 score (Output Validation)

This script is self-contained: run it directly to reproduce every step
described in the project brief.

    python iris_classification.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

RANDOM_STATE = 42
OUTPUT_DIR = "."


# ---------------------------------------------------------------------------
# STEP 1 — INPUT: Load and understand the dataset
# ---------------------------------------------------------------------------
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("=" * 60)
    print("STEP 1: RAW MATERIAL — THE IRIS BENCHMARK")
    print("=" * 60)
    print(f"Samples   : {df.shape[0]}  (balanced: 50 per class)")
    print(f"Features  : {iris.feature_names}")
    print(f"Classes   : {list(iris.target_names)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nClass balance:")
    print(df["species"].value_counts())
    print()

    return iris.data, iris.target, iris.feature_names, iris.target_names, df


# ---------------------------------------------------------------------------
# STEP 2 — PROCESS (a): The Gatekeeper Rule — Feature Scaling
# ---------------------------------------------------------------------------
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on training data
    X_test_scaled = scaler.transform(X_test)         # reuse the same transform
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# STEP 3 — PROCESS (b): Structural Integrity — The Train/Test Split
# ---------------------------------------------------------------------------
def split_data(X, y, test_size=0.2):
    print("=" * 60)
    print("STEP 2: STRUCTURAL INTEGRITY — THE SPLIT")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,       # randomize before splitting to remove order bias
        stratify=y,         # keep class balance identical in both sets
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set    : {X_test.shape[0]} samples")
    print()
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# STEP 4 — Tuning the Engine: choosing the optimal K (the "elbow")
# ---------------------------------------------------------------------------
def find_optimal_k(X_train, y_train, k_range=range(1, 21), cv_folds=5):
    """
    Choose K using cross-validation on the TRAINING data only.
    The test set stays completely untouched until final evaluation —
    picking K by peeking at the test set would be data leakage, and would
    also tend to pick k=1 (the noisy/overfit end of the curve shown on the
    'Tuning the Engine' slide) just by chance on a small test set.
    """
    print("=" * 60)
    print("STEP 3: TUNING THE ENGINE — CHOOSING 'K'")
    print("=" * 60)

    error_rates = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
        error_rates.append(1 - scores.mean())

    optimal_k = k_range[int(np.argmin(error_rates))]
    print(f"Optimal K found via {cv_folds}-fold cross-validation: k = {optimal_k}")
    print()

    # Plot error rate vs K (mirrors the "Tuning the Engine" slide)
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), error_rates, marker="o", linestyle="--", color="#1b4965")
    plt.scatter(
        [optimal_k],
        [error_rates[optimal_k - k_range[0]]],
        color="#d64933",
        s=150,
        zorder=5,
        label=f"Optimal K = {optimal_k}",
    )
    plt.title(f"Choosing K: {cv_folds}-Fold CV Error Rate vs. K Value")
    plt.xlabel("K Value")
    plt.ylabel("Error Rate")
    plt.xticks(list(k_range))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/elbow_plot.png", dpi=150)
    plt.close()

    return optimal_k


# ---------------------------------------------------------------------------
# STEP 5 — PROCESS (c): The Workflow — instantiate / fit / predict
# ---------------------------------------------------------------------------
def train_knn(X_train, y_train, k):
    print("=" * 60)
    print("STEP 4: THE WORKFLOW — SCIKIT-LEARN")
    print("=" * 60)
    model = KNeighborsClassifier(n_neighbors=k)   # INSTANTIATE
    model.fit(X_train, y_train)                   # FIT (memorize the map)
    print(f"Model trained: KNeighborsClassifier(n_neighbors={k})")
    print()
    return model


# ---------------------------------------------------------------------------
# STEP 6 — OUTPUT: Validation — Confusion Matrix + F1 Score
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, target_names):
    print("=" * 60)
    print("STEP 5: OUTPUT VALIDATION")
    print("=" * 60)

    predictions = model.predict(X_test)            # PREDICT (apply logic)

    acc = accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average="macro")
    f1_weighted = f1_score(y_test, predictions, average="weighted")

    print(f"Accuracy         : {acc:.4f}")
    print(f"F1 Score (macro) : {f1_macro:.4f}")
    print(f"F1 Score (weight): {f1_weighted:.4f}\n")

    print("Full classification report (precision / recall / F1 per class):")
    print(classification_report(y_test, predictions, target_names=target_names))

    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix (rows = actual, cols = predicted):")
    print(cm)
    print()

    # Plot the confusion matrix (mirrors "The Diagnostic Tool" slide)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=False,
    )
    plt.title(f"Confusion Matrix — Accuracy: {acc:.2%}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    return predictions, acc, f1_macro


# ---------------------------------------------------------------------------
# MAIN — The Full Architecture (Project 2 Pipeline)
# ---------------------------------------------------------------------------
def main():
    X, y, feature_names, target_names, df = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    optimal_k = find_optimal_k(X_train_scaled, y_train)
    model = train_knn(X_train_scaled, y_train, k=optimal_k)
    evaluate_model(model, X_test_scaled, y_test, target_names)

    print("=" * 60)
    print("PIPELINE COMPLETE — Project 2 milestone achieved.")
    print("Artifacts saved: elbow_plot.png, confusion_matrix.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
