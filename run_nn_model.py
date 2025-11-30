import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import KFold
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score
)

from models import load_data, feature_selection

def threshold_cross_validation(model, X, y, device, thresholds, k=5):

    # Convert to numpy for easy indexing
    X_np = X.values
    y_np = y.values

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Store metrics for each threshold
    scores = {t: [] for t in thresholds}

    model.eval()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_np)):
        print(f"\n=== Fold {fold+1}/{k} ===")

        X_val = torch.tensor(X_np[val_idx], dtype=torch.float32).to(device)
        y_val = y_np[val_idx]

        # Get logits for this fold
        with torch.no_grad():
            logits = model(X_val).cpu().numpy().flatten()
            probs = 1 / (1 + np.exp(-logits))

        # Test each threshold on THIS fold
        for t in thresholds:
            preds = (probs >= t).astype(int)

            f1 = f1_score(y_val, preds)
            scores[t].append(f1)

            print(f"threshold={t:.2f} | F1={f1:.4f}")

    # Compute average F1 per threshold
    avg_scores = {t: np.mean(scores[t]) for t in thresholds}

    print("\n=== Cross-validated Threshold Results ===")
    for t in thresholds:
        print(f"threshold={t:.2f} | avg F1={avg_scores[t]:.4f}")

    # Pick best threshold
    best_t = max(avg_scores, key=lambda t: avg_scores[t])
    print(f"\n>>> BEST THRESHOLD = {best_t:.2f} (Avg F1={avg_scores[best_t]:.4f})")

    return best_t, avg_scores


# -------------------------------
# 1. IMPORT YOUR MLP ARCHITECTURE
# -------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super().__init__()
        layers = []
        prev = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h

        layers.append(nn.Linear(prev, 1))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------------
# 2. RUN MODEL ON A GIVEN DATASET
# -------------------------------
def run_model(model_path, X, y, threshold, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model
    model = MLP(input_dim=X.shape[1], hidden_sizes=[64, 32]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Tensor conversion
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(X_tensor).cpu().numpy().flatten()
        probs = 1 / (1 + np.exp(-logits))

    preds = (probs >= threshold).astype(int)

    # Metrics
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, probs)
    auprc = average_precision_score(y, probs)
    cm = confusion_matrix(y, preds)

    print("\n=== Neural Network Evaluation ===")
    print("Threshold:", threshold)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print(f"AUPRC    : {auprc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    return probs, preds

def main():

    print("=== Loading feature-selected datasets from disk ===")

    # Load FS data
    X_train_fs = pd.read_parquet("Data/X_train_fs.parquet")
    X_test_fs  = pd.read_parquet("Data/X_test_fs.parquet")
    y_train    = pd.read_parquet("Data/y_train.parquet").squeeze()
    y_test     = pd.read_parquet("Data/y_test.parquet").squeeze()

    print("Train FS shape:", X_train_fs.shape)
    print("Test FS shape :", X_test_fs.shape)

    # Load selected feature list
    with open("Data/selected_features.txt", "r") as f:
        selected_features = [line.strip() for line in f.readlines()]
    print("Loaded selected features.")

    # ---------------------------
    # DEFINE DEVICE
    # ---------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------
    # LOAD TRAINED MODEL
    # ---------------------------
    print("\nLoading trained model...")
    model = MLP(input_dim=X_train_fs.shape[1], hidden_sizes=[64, 32]).to(device)
    model.load_state_dict(torch.load("best_mlp_64_32_posweight_2.0.pth", map_location=device))
    model.eval()

    print("Model loaded.")

    # ---------------------------
    # THRESHOLD VALUES TO TEST
    # ---------------------------
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    # ---------------------------
    # RUN CROSS-VALIDATION FOR THRESHOLD
    # ---------------------------
    best_threshold, scores = threshold_cross_validation(
        model=model,
        X=X_train_fs,
        y=y_train,
        device=device,
        thresholds=thresholds,
        k=5
    )

    print("\nBest threshold from CV:", best_threshold)

    # ---------------------------
    # FINAL EVALUATION ON TEST SET
    # ---------------------------
    run_model(
        model_path="best_mlp_64_32_posweight_2.0.pth",
        X=X_test_fs,
        y=y_test,
        threshold=.55,   # <-- USE BEST THRESHOLD
        device=device
    )


if __name__ == "__main__":
    main()
