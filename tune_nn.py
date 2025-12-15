import torch
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from models import MLP, train_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# SIMPLE VALIDATION EVALUATION

def evaluate_val(model, X_val, y_val, device, threshold=0.60):

    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(X_val.values, dtype=torch.float32).to(device)
        ).cpu().numpy().flatten()

    probs = 1 / (1 + torch.tensor(logits).exp().neg().add(1).reciprocal().numpy())  # sigmoid
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds)
    rec = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    auc = roc_auc_score(y_val, probs)
    cm = confusion_matrix(y_val, preds)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(cm)

    return f1


# MAIN — TUNE pos_weight

def main():

    print("\n=== Loading feature-selected train data ===")

    X_train = pd.read_parquet("Data/X_train_fs.parquet")
    y_train = pd.read_parquet("Data/y_train.parquet").squeeze()

    # Split into TRAIN and VALIDATION
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.10,
        stratify=y_train,
        random_state=42
    )

    # Convert to Torch
    train_ds = TensorDataset(
        torch.tensor(X_tr.values, dtype=torch.float32),
        torch.tensor(y_tr.values, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

    # Values to test
    pos_weights = [1.0, 1.5, 2.0]

    best_f1 = -1
    best_pw = None
    best_model = None

    for pw in pos_weights:
        print("\n" + "=" * 50)
        print(f"Training MLP (64,32) with pos_weight = {pw}")
        print("=" * 50)

        model = MLP(input_dim=X_train.shape[1], hidden_sizes=[64, 32]).to(DEVICE)

        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            poss_weight=pw
        )

        print("\n--- Validation Performance ---")
        f1 = evaluate_val(trained_model, X_val, y_val, DEVICE)

        if f1 > best_f1:
            best_f1 = f1
            best_pw = pw
            best_model = trained_model

    # Print final decision
    print("\n=====================================")
    print(f"BEST pos_weight = {best_pw}")
    print(f"BEST Validation F1 = {best_f1:.4f}")
    print("=====================================")

    # Save the best model
    save_path = f"best_mlp_64_32_posweight_{best_pw}.pth"
    torch.save(best_model.state_dict(), save_path)
    print(f"Saved best model to {save_path}")


if __name__ == "__main__":
    main()
