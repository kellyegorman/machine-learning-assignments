import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("hw-3/wine_dataset.csv")
df["style"] = df["style"].map({"red": 0, "white": 1})
X = df.drop("style", axis=1)
y = df["style"]

X_train, X_test, y_train, y_test = train_test_split(
    #70/30
    X, y, test_size=0.3, random_state=1 
)

# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

results = []
# no regularization
model_none = LogisticRegression(penalty=None, max_iter=5000)
model_none.fit(X_train, y_train)
y_pred = model_none.predict(X_test)
acc_none = accuracy_score(y_test, y_pred)
norm_none = np.linalg.norm(model_none.coef_)
zeros_none = np.sum(model_none.coef_ == 0)
results.append(("none", "N/A", acc_none, norm_none, zeros_none, model_none))

# L1 models
l1_models = []
for C in [0.01, 0.1, 1]:
    model = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    norm = np.linalg.norm(model.coef_)
    zeros = np.sum(model.coef_ == 0)
    results.append(("l1", C, acc, norm, zeros, model))
    l1_models.append((C, acc, norm, zeros, model))

# L2 models
l2_models = []
for C in [0.01, 0.1, 1]:
    model = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    norm = np.linalg.norm(model.coef_)
    zeros = np.sum(model.coef_ == 0)
    results.append(("l2", C, acc, norm, zeros, model))
    l2_models.append((C, acc, norm, zeros, model))

print("All model accuracies:")
for penalty, C, acc, norm, zeros, model in results:
    print(f"penalty={penalty}, C={C}, accuracy={acc:.4f}")

#No regularization - L2 norm
print("\nNo regularization model:")
print(f"Accuracy = {acc_none:.4f}")
print(f"L2 norm of weights = {norm_none:.4f}")
print(f"Number of zero weights = {zeros_none}")

#Best L1 
best_l1 = max(l1_models, key=lambda x: x[1])
best_l1_C, best_l1_acc, best_l1_norm, best_l1_zeros, best_l1_model = best_l1

print("\nBest L1 model:")
print(f"C = {best_l1_C}")
print(f"Accuracy = {best_l1_acc:.4f}")
print(f"L2 norm of weights = {best_l1_norm:.4f}")
print(f"Number of zero weights = {best_l1_zeros}")

if best_l1_norm > norm_none:
    print("L1 norm is higher than no regularization.")
else:
    print("L1 norm is lower than no regularization.")

# Best L2
best_l2 = max(l2_models, key=lambda x: x[1])
best_l2_C, best_l2_acc, best_l2_norm, best_l2_zeros, best_l2_model = best_l2

print("\nBest L2 model:")
print(f"C = {best_l2_C}")
print(f"Accuracy = {best_l2_acc:.4f}")
print(f"L2 norm of weights = {best_l2_norm:.4f}")
print(f"Number of zero weights = {best_l2_zeros}")

if best_l2_norm > norm_none:
    print("L2 norm is higher than no regularization.")
else:
    print("L2 norm is lower than no regularization.")

if best_l2_norm > best_l1_norm:
    print("L2 norm is higher than best L1.")
else:
    print("L2 norm is lower than best L1.")

# zero weights in the 3 selected models
print("\nZero weights in selected models:")
print(f"No regularization: {zeros_none}")
print(f"Best L1: {best_l1_zeros}")
print(f"Best L2: {best_l2_zeros}")
