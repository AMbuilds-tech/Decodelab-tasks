# Project 2 — Data Classification Using AI

DecodeLabs Industrial Training Kit, Batch 2026

## What this delivers

A complete, runnable supervised-learning pipeline that satisfies every
requirement in the Project 2 brief:

| Brief requirement | Where it's handled |
|---|---|
| Load and understand a dataset | `load_data()` — loads the Iris dataset (150 samples, 3 classes, 4 features) |
| Split data into training/testing sets | `split_data()` — 80/20 stratified, shuffled split |
| Apply a simple classification algorithm | K-Nearest Neighbors via `train_knn()` |
| Feature scaling ("Gatekeeper Rule") | `scale_features()` — `StandardScaler`, fit only on training data |
| Choosing K ("Tuning the Engine") | `find_optimal_k()` — 5-fold cross-validation on the *training* set only (avoids leaking the test set into model selection), saved as `elbow_plot.png` |
| Output validation | `evaluate_model()` — confusion matrix (`confusion_matrix.png`), precision/recall/F1 per class, and overall accuracy/F1 |

## How to run

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
python iris_classification.py
```

## Results from the included run

- Optimal K (via cross-validation): **5**
- Test accuracy: **93.3%**
- Macro F1 score: **0.933**
- Only confusion: 2 *virginica* samples predicted as *versicolor* (these two
  species overlap slightly in petal measurements — a well-known property of
  this dataset, visible in the "Architectural Paradigms" slide).

## A note on going further (per the "Conclusion" slide)

The brief encourages experimenting rather than stopping at "it works." Easy
extensions to try:

- Swap `KNeighborsClassifier` for `LogisticRegression`, `DecisionTreeClassifier`,
  or `SVC` and compare F1 scores on the same split.
- Test the trained model on a few hand-made flower measurements outside the
  dataset to see how it generalizes.
- Try `test_size=0.3` or a different `random_state` and see how much the
  reported accuracy moves — a useful lesson in why a single train/test split
  can be a bit noisy on only 150 samples.
