# kryptonite_nn_train.py
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump

# ----------------------------
# Configuration
# ----------------------------
# n = 10   # replace with actual dimension, e.g., 10, 12, 14, etc.

for n in range(10 ,22, 2):
    base = r"C:\Users\sandr\OneDrive\Documents\MML\Kryptonite-2\Datasets"
    train_X_path = os.path.join(base, f"kryptonite-{n}-X.npy")
    train_y_path = os.path.join(base, f"kryptonite-{n}-y.npy")
    hidden_X_path = os.path.join(base, f"hidden-kryptonite-{n}-X.npy")

    out_dir = 'hiddenlabels'
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------
    # Load and preprocess
    # ----------------------------
    print("Current working directory:", os.getcwd())
    X = np.load(train_X_path)
    y = np.load(train_y_path)

    # Split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ----------------------------
    # Define candidate architectures
    # ----------------------------
    architectures = {
        "Small MLP": (128,),
        "Medium MLP": (256, 128),
        "Large MLP": (512, 256, 128)
    }

    results = {}

    for name, hidden_layers in architectures.items():
        print(f"\n=== Training {name} ===")
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=1e-4,              # L2 regularization (weight decay)
            learning_rate_init=1e-3,
            batch_size=256,
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10,
            verbose=True
        )

        mlp.fit(X_train, y_train)

        # Evaluate
        train_acc = mlp.score(X_train, y_train)
        val_acc = mlp.score(X_val, y_val)
        test_acc = mlp.score(X_test, y_test)
        results[name] = (train_acc, val_acc, test_acc)

        print(f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

        # Save model
        dump(mlp, f"{name.replace(' ','_')}_model_{n}.joblib")

    # ----------------------------
    # Summarize performance
    # ----------------------------
    print("\n=== Summary of Neural Network Results ===")
    for name, (train_acc, val_acc, test_acc) in results.items():
        print(f"{name:15s} | Train: {train_acc:.3f} | Val: {val_acc:.3f} | Test: {test_acc:.3f}")

    # Pick best model by validation accuracy
    best_model_name = max(results, key=lambda k: results[k][1])
    print(f"\nBest performing model: {best_model_name}")
    best_model_path = f"{best_model_name.replace(' ','_')}_model_{n}.joblib"
    best_model = MLPClassifier()
    from joblib import load
    best_model = load(best_model_path)

    # ----------------------------
    # Evaluate in detail on test set
    # ----------------------------
    y_test_pred = best_model.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, digits=4))

    # ----------------------------
    # Learning curve (optional)
    # ----------------------------
    plt.figure(figsize=(6,4))
    plt.plot(best_model.loss_curve_, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for {best_model_name}, n = {n}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'loss_curve_{best_model_name.replace(" ","_")}_n_{n}.png', dpi=200)

    # ----------------------------
    # Predict hidden labels
    # ----------------------------
    X_hidden = np.load(hidden_X_path)
    X_hidden = scaler.transform(X_hidden)
    y_hidden_pred = best_model.predict(X_hidden)
    np.save(os.path.join(out_dir, f"y_predicted_{n}.npy"), y_hidden_pred.astype(np.uint8))

    print(f"\nSaved hidden predictions → {out_dir}/y_predicted_{n}.npy")
