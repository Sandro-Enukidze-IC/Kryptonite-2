# ExpDes_kfold.py
import os
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------
# Configuration
# -----------------------
BASE = r"C:\Users\sandr\OneDrive\Documents\MML\Kryptonite-2\Datasets"
OUT_DIR = "hiddenlabels"
os.makedirs(OUT_DIR, exist_ok=True)

# Which dataset dimensions to run
n_values = list(range(16, 22, 2))   # 10,12,...,20

# Cross-validation settings
k_folds = 5
n_repeats = 3            # how many repeated k-fold runs (use >1 for more robust uncertainty)
val_fraction_within_train = 0.2  # when we take the training portion, carve this out as an external val set

# Experiment seeds base (we'll vary by repeat+fold)
seed_base = 42

# Architectures to evaluate
architectures = {
    "Small MLP": (50,),
    "Medium MLP": (200, 100),
    "Large MLP": (400, 200, 100)
}

# MLP hyperparameters (you can tune these)
mlp_params = dict(
    activation='relu',
    solver='adam',
    alpha=1e-2,
    learning_rate_init=1e-4,
    batch_size=128,
    max_iter=200,
    early_stopping=True,     # we do explicit val set so turn off internal early stopping
    validation_fraction = 0.20,
    n_iter_no_change=20,

    verbose=False
)

# -----------------------
# Helper: run single fold
# -----------------------
def run_single_fold(X, y, train_idx, test_idx, hidden_X, arch, repeat_seed, fold_id):
    """
    train_idx/test_idx are indices for the StratifiedKFold split.
    We will further split train_idx into train/val (external val).
    Returns (train_acc, val_acc, test_acc, trained_model)
    """
    # Build train / temp / test from indices
    X_train_full = X[train_idx]
    y_train_full = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    # # carve out external validation from X_train_full
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_train_full, y_train_full,
    #     test_size=val_fraction_within_train,
    #     stratify=y_train_full,
    #     random_state=seed_base + repeat_seed + fold_id
    # )
    X_train = X_train_full
    y_train = y_train_full


    # scale based ONLY on X_train
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = None
    X_test_s = scaler.transform(X_test)
    if hidden_X is not None:
        X_hidden_s = scaler.transform(hidden_X)
    else:
        X_hidden_s = None

    # instantiate model with fold-specific seed for repeatability
    mlp = MLPClassifier(hidden_layer_sizes=arch, random_state=seed_base + repeat_seed + fold_id, **mlp_params)

    # train
    mlp.fit(X_train_s, y_train)

    # evaluate
    train_acc = accuracy_score(y_train, mlp.predict(X_train_s))
    val_acc = np.nan
    test_acc = accuracy_score(y_test, mlp.predict(X_test_s))

    # predict hidden if provided
    hidden_preds = None
    if X_hidden_s is not None:
        hidden_preds = mlp.predict(X_hidden_s)

    return train_acc, val_acc, test_acc, mlp, scaler, hidden_preds

# -----------------------
# Main experiment loop
# -----------------------
for n in n_values:
    print(f"\n\n===== Running experiments for n = {n} =====")
    train_X_path = os.path.join(BASE, f"kryptonite-{n}-X.npy")
    train_y_path = os.path.join(BASE, f"kryptonite-{n}-y.npy")
    hidden_X_path = os.path.join(BASE, f"hidden-kryptonite-{n}-X.npy")

    # load data (if y missing, raise helpful error)
    try:
        X = np.load(train_X_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {train_X_path}")
    try:
        y = np.load(train_y_path)
    except Exception as e:
        raise FileNotFoundError(f"Could not find training labels {train_y_path}. If you do not have labels, you cannot compute CV accuracies. Error: {e}")

    # try load hidden (may exist)
    hidden_X = None
    if os.path.exists(hidden_X_path):
        hidden_X = np.load(hidden_X_path)

    # storage for results: arch -> list of fold results over repeats
    metrics = {name: {"train": [], "val": [], "test": [], "models": [], "scalers": []} for name in architectures.keys()}

    # Repeat the stratified k-fold several times (different shuffles)
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed_base + repeat)
        for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"n={n} repeat={repeat+1}/{n_repeats} fold={fold_id+1}/{k_folds}")
            for name, arch in architectures.items():
                tr_acc, v_acc, te_acc, model, scaler, hidden_preds = run_single_fold(
                    X, y, train_idx, test_idx, hidden_X, arch, repeat, fold_id
                )
                metrics[name]["train"].append(tr_acc)
                metrics[name]["val"].append(v_acc)
                metrics[name]["test"].append(te_acc)
                metrics[name]["models"].append(model)
                metrics[name]["scalers"].append(scaler)

    # After repeats and folds compute mean and std
    summary = {}
    for name in architectures.keys():
        train_arr = np.array(metrics[name]["train"])
        val_arr   = np.array(metrics[name]["val"])
        test_arr  = np.array(metrics[name]["test"])

        summary[name] = {
            "train_mean": train_arr.mean(), "train_std": train_arr.std(ddof=1),
            "val_mean": val_arr.mean(), "val_std": val_arr.std(ddof=1),
            "test_mean": test_arr.mean(), "test_std": test_arr.std(ddof=1),
            "n_runs": len(test_arr)
        }

    # print summary table for this n
    print("\n=== Summary (mean ± std) for n =", n)
    for name, s in summary.items():
        print(f"{name:12s} | Train: {s['train_mean']:.4f} ± {s['train_std']:.4f} | "
              f"Val: {s['val_mean']:.4f} ± {s['val_std']:.4f} | "
              f"Test: {s['test_mean']:.4f} ± {s['test_std']:.4f} | runs: {s['n_runs']}")

    # choose best architecture by mean validation accuracy
    best_arch_name = max(summary.keys(), key=lambda k: summary[k]["val_mean"])
    print(f"\nBest architecture by mean val acc: {best_arch_name}")

    # find the model instance with highest val acc among stored ones for best_arch_name
    best_idx = np.argmax(metrics[best_arch_name]["val"])
    best_model = metrics[best_arch_name]["models"][best_idx]
    best_scaler = metrics[best_arch_name]["scalers"][best_idx]

    # Evaluate chosen model in detail on a single hold-out test fold (we have test arrays stored separately per fold)
    # To produce confusion matrix similar to what you did before, recompute test split from that fold
    # We can recompute using the fold that generated best_idx:
    total_folds = k_folds * n_repeats
    fold_of_best = best_idx % k_folds
    repeat_of_best = best_idx // k_folds
    # Recreate the exact split that produced that best model to extract its test set
    skf_best = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed_base + repeat_of_best)
    for fid, (train_idx, test_idx) in enumerate(skf_best.split(X, y)):
        if fid == fold_of_best:
            # reconstruct the same train/val/test used earlier
            X_train_full = X[train_idx]
            y_train_full = y[train_idx]
        
            X_test = X[test_idx]
            y_test = y[test_idx]
            break

    # scale and compute confusion matrix on that test fold
    X_test_s = best_scaler.transform(X_test)
    y_test = y[test_idx]
    y_test_pred = best_model.predict(X_test_s)
    print("\nConfusion Matrix for best model (one representative fold):")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, digits=4))

    # Save predictions for the hidden set using the best model/scaler pair (if hidden present)
    if hidden_X is not None:
        X_hidden_s = best_scaler.transform(hidden_X)
        hidden_preds = best_model.predict(X_hidden_s)
        out_path = os.path.join(OUT_DIR, f"y_predicted_{n}.npy")
        np.save(out_path, hidden_preds.astype(np.uint8))
        print(f"\nSaved hidden predictions -> {out_path}")

    # Save summary JSON-like arrays (you can also dump to json file)
    # Create plot: bar chart of mean test accuracies with error bars
    arch_names = list(summary.keys())
    test_means = [summary[a]["test_mean"] for a in arch_names]
    test_stds  = [summary[a]["test_std"] for a in arch_names]

    plt.figure(figsize=(8,5))
    x = np.arange(len(arch_names))
    plt.bar(x, test_means, yerr=test_stds, capsize=8)
    plt.xticks(x, arch_names)
    plt.ylim(0.5, 1.0)
    plt.ylabel("Test Accuracy")
    plt.title(f"Test Accuracy (mean ± std) over {k_folds}-fold CV, repeats={n_repeats}, n={n}")
    for i, v in enumerate(test_means):
        plt.text(i, v + 0.005, f"{v:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig(f"accuracy_bar_n_{n}.png", dpi=200)
    plt.close()

    # Also save numeric summary to file for later reporting
    np.savez_compressed(f"summary_n_{n}.npz", **{name: summary[name] for name in summary})
    print(f"Saved summary and plot for n={n}")

print("\nAll experiments complete.")
