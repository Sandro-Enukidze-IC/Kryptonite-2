# ExpDes_kfold_knn.py
import os
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------
# Configuration
# -----------------------
BASE = r"C:\Users\sandr\OneDrive\Documents\MML\Kryptonite-2\Datasets"
OUT_DIR = "hiddenlabels"
os.makedirs(OUT_DIR, exist_ok=True)
MODELS_DIR = "trained_models_knn"
os.makedirs(MODELS_DIR, exist_ok=True)

# Which dataset dimensions to run (change to the n values you actually have)
n_values = [10]   # example: [10,12,14,...] - keep only those you have files for

# Cross-validation settings
k_folds = 5
n_repeats = 3            # number of repeated k-fold runs
val_fraction_within_train = 0.2  # carve an external val set from each training fold

seed_base = 42

# kNN hyperparameter grid to evaluate
k_list = [1, 3, 5, 7, 11]
weights_list = ['uniform', 'distance']
metric = 'minkowski'  # default; can set p=2 for Euclidean later when instantiating

# Storage for plot aesthetics
plt.rcParams.update({'figure.max_open_warning': 0})

# -----------------------
# Helper: run single fold for a given hyperparameter combination
# -----------------------
def run_single_fold_knn(X, y, train_idx, test_idx, hidden_X, n_neighbors, weights, repeat_seed, fold_id):
    """
    Train and evaluate a KNeighborsClassifier on a single fold.
    Returns (train_acc, val_acc, test_acc, model, scaler, hidden_preds)
    """
    X_train_full = X[train_idx]
    y_train_full = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    # carve out external validation from the training portion
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_fraction_within_train,
        stratify=y_train_full,
        random_state=seed_base + repeat_seed + fold_id
    )

    # Scale based on X_train only (important for distance-based KNN)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    X_hidden_s = scaler.transform(hidden_X) if (hidden_X is not None) else None

    # instantiate KNN (use Euclidean p=2)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric='minkowski', p=2, n_jobs=-1)

    # fit (kNN "training" is storing the scaled training data)
    knn.fit(X_train_s, y_train)

    # evaluate
    train_acc = accuracy_score(y_train, knn.predict(X_train_s))
    val_acc   = accuracy_score(y_val, knn.predict(X_val_s))
    test_acc  = accuracy_score(y_test, knn.predict(X_test_s))

    hidden_preds = None
    if X_hidden_s is not None:
        hidden_preds = knn.predict(X_hidden_s)

    return train_acc, val_acc, test_acc, knn, scaler, hidden_preds

# -----------------------
# Main experiment loop
# -----------------------
for n in n_values:
    print(f"\n\n===== Running kNN experiments for n = {n} =====")
    train_X_path = os.path.join(BASE, f"kryptonite-{n}-X.npy")
    train_y_path = os.path.join(BASE, f"kryptonite-{n}-y.npy")
    hidden_X_path = os.path.join(BASE, f"hidden-kryptonite-{n}-X.npy")

    # load data
    if not os.path.exists(train_X_path):
        raise FileNotFoundError(f"Missing features: {train_X_path}")
    if not os.path.exists(train_y_path):
        raise FileNotFoundError(f"Missing labels: {train_y_path} — cannot run supervised CV without labels.")
    X = np.load(train_X_path)
    y = np.load(train_y_path)
    hidden_X = np.load(hidden_X_path) if os.path.exists(hidden_X_path) else None

    # Prepare grid of hyperparameters (tuples)
    param_grid = [(k, w) for k in k_list for w in weights_list]
    print(f"Evaluating hyperparams (n_neighbors, weights): {param_grid}")

    # storage: param_key -> lists over (folds * repeats)
    metrics = { (k,w): {"train": [], "val": [], "test": [], "models": [], "scalers": []} for (k,w) in param_grid }

    # Repeat stratified k-fold CV
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed_base + repeat)
        for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"n={n} repeat={repeat+1}/{n_repeats} fold={fold_id+1}/{k_folds}")
            for (k_v, w_v) in param_grid:
                tr_acc, v_acc, te_acc, model, scaler, hidden_preds = run_single_fold_knn(
                    X, y, train_idx, test_idx, hidden_X, k_v, w_v, repeat, fold_id
                )
                metrics[(k_v, w_v)]["train"].append(tr_acc)
                metrics[(k_v, w_v)]["val"].append(v_acc)
                metrics[(k_v, w_v)]["test"].append(te_acc)
                metrics[(k_v, w_v)]["models"].append(model)
                metrics[(k_v, w_v)]["scalers"].append(scaler)

    # Summarize mean ± std for each param combo
    summary = {}
    for key, vals in metrics.items():
        train_arr = np.array(vals["train"])
        val_arr   = np.array(vals["val"])
        test_arr  = np.array(vals["test"])
        summary[key] = {
            "train_mean": train_arr.mean(), "train_std": train_arr.std(ddof=1) if len(train_arr)>1 else 0.0,
            "val_mean": val_arr.mean(),     "val_std": val_arr.std(ddof=1) if len(val_arr)>1 else 0.0,
            "test_mean": test_arr.mean(),   "test_std": test_arr.std(ddof=1) if len(test_arr)>1 else 0.0,
            "n_runs": len(test_arr)
        }

    # Print summary table
    print("\n=== Summary (mean ± std) for n =", n)
    for key, s in sorted(summary.items(), key=lambda kv: -kv[1]["val_mean"]):
        k_v, w_v = key
        print(f"k={k_v:2d}, weights={w_v:8s} | Train: {s['train_mean']:.4f} ± {s['train_std']:.4f} | "
              f"Val: {s['val_mean']:.4f} ± {s['val_std']:.4f} | Test: {s['test_mean']:.4f} ± {s['test_std']:.4f} | runs: {s['n_runs']}")

    # Choose best hyperparam by mean validation accuracy
    best_key = max(summary.keys(), key=lambda k: summary[k]["val_mean"])
    print(f"\nBest hyperparameters by mean val acc: k={best_key[0]}, weights={best_key[1]}")

    # Find best single model instance (highest per-fold val acc for that hyperparam)
    best_vals = metrics[best_key]["val"]
    best_idx = int(np.argmax(best_vals))
    best_model = metrics[best_key]["models"][best_idx]
    best_scaler = metrics[best_key]["scalers"][best_idx]

    # Reconstruct test split corresponding to best_idx to show confusion matrix
    # best_idx indexes into the flattened list of runs in the order we appended:
    # order: for repeat in 0..n_repeats-1: for fold in 0..k_folds-1: appended -> so idx = repeat*k_folds + fold
    repeat_of_best = best_idx // k_folds
    fold_of_best = best_idx % k_folds
    skf_best = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed_base + repeat_of_best)
    for fid, (train_idx, test_idx) in enumerate(skf_best.split(X, y)):
        if fid == fold_of_best:
            # reconstruct the external val split used during that run
            X_train_full = X[train_idx]; y_train_full = y[train_idx]
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full,
                test_size=val_fraction_within_train,
                stratify=y_train_full,
                random_state=seed_base + repeat_of_best + fold_of_best
            )
            X_test = X[test_idx]
            break

    # Compute confusion matrix on that representative test fold
    X_test_s = best_scaler.transform(X_test)
    y_test = y[test_idx]
    y_test_pred = best_model.predict(X_test_s)
    print("\nConfusion Matrix for best (representative) run:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, digits=4))

    # Save best model and scaler
    model_name = f"knn_best_n{n}_k{best_key[0]}_{best_key[1]}.joblib"
    dump({"model": best_model, "scaler": best_scaler, "params": {"n": n, "k": best_key[0], "weights": best_key[1]}},
         os.path.join(MODELS_DIR, model_name))
    print(f"\nSaved best model+scaler -> {os.path.join(MODELS_DIR, model_name)}")

    # Save hidden predictions with the chosen best model/scaler
    if hidden_X is not None:
        X_hidden_s = best_scaler.transform(hidden_X)
        hidden_preds = best_model.predict(X_hidden_s)
        out_path = os.path.join(OUT_DIR, f"y_predicted_{n}.npy")
        np.save(out_path, hidden_preds.astype(np.uint8))
        print(f"Saved hidden predictions -> {out_path}")

    # Plot: bar chart of mean test accuracies (by hyperparam) with error bars ± std
    labels = [f"k={k},w={w}" for (k,w) in summary.keys()]
    test_means = [summary[(k,w)]["test_mean"] for (k,w) in summary.keys()]
    test_stds  = [summary[(k,w)]["test_std"]  for (k,w) in summary.keys()]

    x = np.arange(len(labels))
    plt.figure(figsize=(10,5))
    plt.bar(x, test_means, yerr=test_stds, capsize=6)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylim(0.0, 1.0)
    plt.ylabel("Test Accuracy")
    plt.title(f"kNN Test Accuracy (mean ± std) over {k_folds}-fold CV, repeats={n_repeats}, n={n}")
    for i, v in enumerate(test_means):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
    plt.tight_layout()
    plot_path = f"knn_accuracy_bar_n_{n}.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved plot -> {plot_path}")

    # Save numeric summary for later (npz)
    np.savez_compressed(f"knn_summary_n_{n}.npz", **{f"{k}_{w}": summary[(k,w)] for (k,w) in summary})
    print(f"Saved numeric summary -> knn_summary_n_{n}.npz")

print("\nAll kNN experiments complete.")
