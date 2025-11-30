import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import joblib


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from scipy.stats import loguniform

from xgboost import XGBClassifier

# GLOBAL VARS
RANDOM_STATE = 42

###########################
# MLP NN PYTORCH FUNCTIONS#
###########################

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super().__init__()
        layers = []
        prev = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))  # fully connected layer
            layers.append(nn.ReLU())  # activation function
            prev = h

        layers.append(nn.Linear(prev, 1))  # final output layer

        self.net = nn.Sequential(*layers)  # build network

    def forward(self, x):
        return self.net(x)

def make_loaders(X_train, y_train, batch_size=2048, val_split=0.1):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train.values,
        y_train.values,
        test_size=val_split,
        stratify=y_train.values,     # <-- CRITICAL
        random_state=RANDOM_STATE
    )

    # Convert to tensors
    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, lr=0.001, weight_decay=1e-4,
                epochs=24, device="cpu", poss_weight=1.5):
    model.to(device)

    pos_weight = torch.tensor([poss_weight]).to(device)  # real-world ratio 80/20
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    patience = 3
    wait = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb.float())
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).squeeze()
                val_loss += criterion(preds, yb.float()).item()

        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model


def evaluate_torch(model, X_test, y_test, device="cpu", name="Torch MLP"):
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_test.values, dtype=torch.float32).to(device))
        logits = logits.cpu().numpy().squeeze()

    # Convert logits → probabilities
    probs = 1 / (1 + np.exp(-logits))  # sigmoid

    preds_label = (probs >= 0.60).astype(int)

    acc = accuracy_score(y_test, preds_label)
    prec = precision_score(y_test, preds_label)
    rec = recall_score(y_test, preds_label)
    f1 = f1_score(y_test, preds_label)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds_label)

    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    return f1

def tune_pytorch_mlp(X_train_fs, y_train, device="cpu"):

    param_grid = [
        {"hidden_sizes": [64],      "lr": 0.001, "weight_decay": 1e-4},
        {"hidden_sizes": [128],     "lr": 0.001, "weight_decay": 1e-4},
        {"hidden_sizes": [64, 32],  "lr": 0.001, "weight_decay": 1e-4},
        {"hidden_sizes": [128, 64], "lr": 0.001, "weight_decay": 1e-4},
    ]

    train_loader, val_loader = make_loaders(X_train_fs, y_train)

    # 👉 extract REAL validation X/y from val_loader
    X_val_list = []
    y_val_list = []

    for X_batch, y_batch in val_loader:
        X_val_list.append(X_batch.cpu())
        y_val_list.append(y_batch.cpu())

    X_val = torch.cat(X_val_list).numpy()
    y_val = torch.cat(y_val_list).numpy()

    X_val_df = pd.DataFrame(X_val, columns=X_train_fs.columns)
    y_val_ser = pd.Series(y_val)


    best_f1 = 0
    best_model = None

    for params in param_grid:
        print(f"\nTesting params: {params}")

        model = MLP(
            input_dim=X_train_fs.shape[1],
            hidden_sizes=params["hidden_sizes"]
        )

        trained = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            epochs=24,
            device=device
        )

        f1 = evaluate_torch(
            trained,
            X_val_df,
            y_val_ser,
            device=device,
            name="MLP Validation"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_model = trained

    return best_model

def find_best_threshold(model, X_val, y_val, device):
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_val.values, dtype=torch.float32).to(device)).cpu().numpy().flatten()
        probs = 1 / (1 + np.exp(-logits))

    thresholds = np.array([0.525, 0.55, 0.575, 0.60])
    best_f1 = -1
    best_t = 0.50

    print("\n=== Threshold Sweep ===")
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f = f1_score(y_val, preds)

        print(f"t={t:.2f} | Precision={precision_score(y_val, preds):.4f} "
              f"Recall={recall_score(y_val, preds):.4f} F1={f:.4f}")

        if f > best_f1:
            best_f1 = f
            best_t = t

    print(f"\n>>> BEST THRESHOLD = {best_t:.2f} (F1={best_f1:.4f})")
    return best_t


def load_data():
    X_train = pd.read_parquet("Data/US_Accidents_Model_X_Train.parquet")
    X_test = pd.read_parquet("Data/US_Accidents_Model_X_Test.parquet")
    y_train = pd.read_parquet("Data/US_Accidents_Model_y_Train.parquet").squeeze()
    y_test = pd.read_parquet("Data/US_Accidents_Model_y_Test.parquet").squeeze()

    # Ensure y is a Series
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test, name="model"):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n=== {name} Test Performance ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def baseline_log_reg():
    return LogisticRegression(
        solver="saga",
        max_iter=2000,
        random_state=RANDOM_STATE
    )


def baseline_rf():
    return RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

def baseline_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(64,),
        max_iter=2000,
        random_state=RANDOM_STATE
    )


def randomized_search(model, param_dist, X_train_small, y_train_small, n_iter, cv):
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
    )

    search.fit(X_train_small, y_train_small)

    print("\nBest Params:", search.best_params_)
    print("Best CV F1 :", search.best_score_)

    return search.best_estimator_


def feature_selection(
        X_train,
        X_test,
        y_train,
        top_n_xgb=25,
        max_fs_sample=400_000
    ):
    print("\n=== FEATURE SELECTION (XGBoost Only) START ===")
    print("Initial Features:", X_train.shape[1])

    # ---------------------------------------------------
    #  XGBoost Feature Importance
    # ---------------------------------------------------
    print("Running XGBoost feature selection...")

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    # Sampling for speed if dataset is huge
    if len(X_train) > max_fs_sample:
        X_fs = X_train.sample(max_fs_sample, random_state=RANDOM_STATE)
        y_fs = y_train.loc[X_fs.index]
        print(f"XGBoost FS using sample of {max_fs_sample} rows")
    else:
        X_fs = X_train
        y_fs = y_train
        print(f"XGBoost FS using all {len(X_fs)} rows")

    # Fit model
    xgb.fit(X_fs, y_fs)

    # Importance extraction
    importances = xgb.feature_importances_
    importance_series = pd.Series(importances, index=X_train.columns)
    importance_series = importance_series.sort_values(ascending=False)

    # Select top N features
    top_features = importance_series.head(top_n_xgb).index.tolist()
    print(f"XGBoost selected top {len(top_features)} features")

    # Reduce original train/test
    X_train_fs = X_train[top_features]
    X_test_fs = X_test[top_features]

    print("Final features:", X_train_fs.shape[1])
    print("Selected features:", top_features)
    print("=== FEATURE SELECTION DONE ===")

    return X_train_fs, X_test_fs, top_features


def main():
    # Load
    X_train, X_test, y_train, y_test = load_data()

    print("Full training set shape:", X_train.shape)
    print("Full test set shape    :", X_test.shape)

    ##############################################
    #         APPLY XGBOOST FEATURE SELECTION    #
    ##############################################

    X_train_fs, X_test_fs, selected_features = feature_selection(
        X_train,
        X_test,
        y_train,
        top_n_xgb=25
    )

    print("\n[INFO] AFTER FEATURE SELECTION:")
    print("Train shape:", X_train_fs.shape)
    print("Test shape :", X_test_fs.shape)


    # small sample for tuning
    X_train_small, _, y_train_small, _ = train_test_split(
        X_train_fs, y_train,
        train_size=200_000,
        stratify=y_train,
        random_state=RANDOM_STATE
    )

    print("\nTuning sample shape:", X_train_small.shape)
    """
    print("\n=== LOGISTIC REGRESSION ===")

    log_base = baseline_log_reg()
    log_base.fit(X_train_small, y_train_small)
    evaluate(log_base, X_test_fs, y_test, "LogReg Baseline")

    # Coarse LR
    coarse_params = {
        "C": loguniform(1e-3, 1e2),
        "class_weight": [None, "balanced"]
    }

    log_coarse = randomized_search(
        baseline_log_reg(),
        coarse_params,
        X_train_small,
        y_train_small,
        n_iter=8,
        cv=2
    )

    # Fine LR
    fine_params = {
        "C": loguniform(1e-2, 1e1),
        "class_weight": [None, "balanced"]
    }

    log_fine = randomized_search(
        log_coarse,
        fine_params,
        X_train_small,
        y_train_small,
        n_iter=10,
        cv=3
    )

    evaluate(log_fine, X_test_pca, y_test, "LogReg Tuned")

    """

    print("\n=== PYTORCH MLP (Final Model) ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -------------------------------------------------
    # STEP 1 — Split TRAIN into TRAIN + VAL
    # -------------------------------------------------
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train_fs, y_train,
        test_size=0.10,
        stratify=y_train,
        random_state=RANDOM_STATE
    )

    # -------------------------------------------------
    # STEP 2 — Create DataLoaders
    # -------------------------------------------------
    train_ds = TensorDataset(
        torch.tensor(X_train_final.values, dtype=torch.float32),
        torch.tensor(y_train_final.values, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_final.values, dtype=torch.float32),
        torch.tensor(y_val_final.values, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

    # -------------------------------------------------
    # STEP 3 — Build Model
    # -------------------------------------------------
    model = MLP(input_dim=X_train_fs.shape[1], hidden_sizes=[64, 32]).to(device)

    # -------------------------------------------------
    # STEP 4 — Train Model
    # -------------------------------------------------
    model = train_model(model, train_loader, val_loader, device=device)

    # -------------------------------------------------
    # STEP 5 — Threshold Tuning on VALIDATION SET
    # -------------------------------------------------
    best_threshold = find_best_threshold(model, X_val_final, y_val_final, device)
    print("Best threshold:", best_threshold)

    # -------------------------------------------------
    # STEP 6 — Save Weights
    # -------------------------------------------------
    torch.save(model.state_dict(), "mlp_64_32_final.pth")
    print("Saved model to mlp_64_32_final.pth")


    print("\n=== Random Forest Baseline ===")

    # --------------------------
    # BASELINE RF
    # --------------------------
    rf_base = baseline_rf()
    rf_base.fit(X_train_small, y_train_small)
    evaluate(rf_base, X_test_fs, y_test, "Random Forest Baseline")

    # Save baseline RF
    joblib.dump(rf_base, "rf_baseline.pkl")
    print("Saved rf_baseline.pkl")

    # --------------------------
    # COARSE SEARCH
    # --------------------------
    rf_coarse_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced"]
    }

    rf_coarse = randomized_search(
        baseline_rf(),
        rf_coarse_params,
        X_train_small, y_train_small,
        n_iter=10, cv=2
    )

    # Save coarse model
    joblib.dump(rf_coarse, "rf_coarse.pkl")
    print("Saved rf_coarse.pkl")

    # --------------------------
    # FINE SEARCH
    # --------------------------
    rf_fine_params = {
        "n_estimators": [200, 300],
        "max_depth": [15, 20, 25],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"]
    }

    rf_fine = randomized_search(
        rf_coarse,
        rf_fine_params,
        X_train_small, y_train_small,
        n_iter=10, cv=3
    )

    # Save fine-tuned RF
    joblib.dump(rf_fine, "rf_tuned.pkl")
    print("Saved rf_tuned.pkl")

    # --------------------------
    # FINAL EVALUATION
    # --------------------------
    evaluate(rf_fine, X_test_fs, y_test, "Random Forest Tuned")


if __name__ == "__main__":
    main()