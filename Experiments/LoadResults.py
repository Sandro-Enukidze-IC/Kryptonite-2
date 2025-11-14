import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------
# 1. Reload model
# -------------------------
model = load("Medium_MLP_model.joblib")

# -------------------------
# 2. Reload your datasets
# -------------------------
# Replace these with your actual paths or arrays used before
X_train = np.load("Datasets/kryptonite_10_X_train.npy")  # if you saved them
y_train = np.load("Datasets/kryptonite_10_y_train.npy")
X_val = np.load("Datasets/kryptonite_10_X_val.npy")
y_val = np.load("Datasets/kryptonite_10_y_val.npy")
X_test = np.load("Datasets/kryptonite_10_X_test.npy")
y_test = np.load("Datasets/kryptonite_10_y_test.npy")

# -------------------------
# 3. Evaluate accuracy
# -------------------------
train_acc = model.score(X_train, y_train)
val_acc = model.score(X_val, y_val)
test_acc = model.score(X_test, y_test)

print(f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

# -------------------------
# 4. Confusion matrix & classification report
# -------------------------
y_test_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
report = classification_report(y_test, y_test_pred, digits=4)

print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
