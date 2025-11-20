import os
import numpy as np
from joblib import dump
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

BASE = r"C:\Users\sandr\OneDrive\Documents\MML\Kryptonite-2\Datasets"
OUT_DIR = "hiddenlabels"
os.makedirs(OUT_DIR, exist_ok=True)

# n_values = list(range(16, 22, 2))
n_values = [20]

param_grid = {
    "mlp__hidden_layer_sizes": [
        (32,),
        (64, 32),
        (256, 128, 64)
    ],
    "mlp__activation" : ['relu','tanh'],#
    "mlp__alpha" : [1e-2,1e-3,1e-4], #
    "mlp__batch_size" : [64,128,256] #
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for n in n_values:
    print(f"\n=== Running GridSearchCV for n={n} ===")

    X = np.load(os.path.join(BASE, f"kryptonite-{n}-X.npy"))
    y = np.load(os.path.join(BASE, f"kryptonite-{n}-y.npy"))

    hidden_path = os.path.join(BASE, f"hidden-kryptonite-{n}-X.npy")
    hidden_X = np.load(hidden_path) if os.path.exists(hidden_path) else None

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            max_iter=200, 
            random_state=42,
            early_stopping=True, 
            n_iter_no_change=20,
            solver="adam",
            learning_rate_init=1e-4,
            verbose=True
            ))
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True
    )

    grid.fit(X, y)

    print("Best Params:", grid.best_params_)
    print("Best CV Accuracy:", grid.best_score_)

    if hidden_X is not None:
        hidden_pred = grid.best_estimator_.predict(hidden_X)
        out_path = os.path.join(OUT_DIR, f"y_predicted_{n}.npy")
        np.save(out_path, hidden_pred.astype(np.uint8))
        print(f"Saved hidden predictions -> {out_path}")

print("\nAll experiments complete.")
