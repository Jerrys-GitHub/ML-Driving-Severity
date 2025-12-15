import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score

# Paths
test_X_path = "Data/X_test_fs.parquet"
test_y_path = "Data/y_test.parquet"
model_paths = {
    "RF Baseline": "rf_baseline.pkl",
    "RF Tuned": "rf_tuned.pkl",
}

# Load data
print("Loading test data...")
X_test = pd.read_parquet(test_X_path)
y_test = pd.read_parquet(test_y_path)

# Ensure y_test is flat
if hasattr(y_test, "values"):
    y_test = y_test.values.ravel()

# Evaluation function
def evaluate_model(model, X_test, y_test, name):
    print(f"\n==== Evaluating {name} ====")

    # Predict probabilities + labels
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print("Confusion Matrix:")
    print(cm)

# Load and evaluate each model
for name, path in model_paths.items():
    print(f"\nLoading {name} from {path}...")
    model = joblib.load(path)
    evaluate_model(model, X_test, y_test, name)

print("\n=== Evaluation Complete ===")
