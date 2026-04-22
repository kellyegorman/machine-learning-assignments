import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# PART A
# Generate a random dataset with 20 samples. Each sample should have two input
# features and 1 output label (0 or 1). (You can simply use the ones from the previous
# question, but make sure to relabel the outputs)

np.random.seed(42)
eta = 0.1
n_samples = 20
n_trees = 10

X = np.random.randn(n_samples, 2)

# create labels from a rule so both classes appear
raw_score = X[:, 0] + 0.8 * X[:, 1] + 0.2 * np.random.randn(n_samples)
y = (raw_score > 0).astype(int)

# Make sure both classes are here
if len(np.unique(y)) < 2:
    y[: n_samples // 2] = 0
    y[n_samples // 2 :] = 1

print("PART A: Random dataset with 20 samples")
df_data = pd.DataFrame(X, columns=["x1", "x2"])
df_data["y"] = y
print(df_data.to_string(index=True))
print()

# F0 = log(p/(1-p))
# & clip p slightly to avoid log(0)
p = np.mean(y)
p = np.clip(p, 1e-10, 1 - 1e-10)
F0 = np.log(p / (1 - p))

print("PART B: Log-odds of the dataset (prediction of tree 0)")
print(f"p = proportion of class 1 = {p:.6f}")
print(f"F0 = log(p / (1-p)) = {F0:.6f}")
print()

# store F(x) (Initially F0 for every sample)
F = np.full(n_samples, F0, dtype=float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# bernoulli GBM residuals 
p_pred = sigmoid(F)
residuals0 = y - p_pred

print("PART C: Residual terms for each training sample")
df_res0 = pd.DataFrame({
    "sample": np.arange(n_samples),
    "y": y,
    "p_pred": np.round(p_pred, 6),
    "residual": np.round(residuals0, 6)
})
print(df_res0.to_string(index=False))
print()

# Store trees and leaf gammas
all_trees = []
all_leaf_gammas = []

for k in range(1, n_trees + 1):
    # current probabilities and residuals
    p_pred = sigmoid(F)
    residuals = y - p_pred
    # train regression tree on residuals
    tree = DecisionTreeRegressor(max_depth=2, random_state=100 + k)
    tree.fit(X, residuals)
    # leaf assignment for each sample
    leaf_ids = tree.apply(X)
    unique_leaves = np.unique(leaf_ids)
    leaf_gamma = {}
    print(f"STEP ({'d' if k == 1 else 'f'}): Decision tree {k}")
    # compute gamma_jk for each leaf j
    for leaf in unique_leaves:
        idx = np.where(leaf_ids == leaf)[0]
        numerator = np.sum(residuals[idx])
        denominator = np.sum(p_pred[idx] * (1 - p_pred[idx]))
        gamma = numerator / denominator if denominator != 0 else 0.0
        leaf_gamma[leaf] = gamma
        print(f"Leaf node {leaf}: gamma_j{k} = {gamma:.6f}")

# store tree and leaf gammas
    all_trees.append(tree)
    all_leaf_gammas.append(leaf_gamma)

    # Extra output for tree 1 required in part (e)
    if k == 1:
        print("PART E")
        # leaf assignment for each sample in tree 1
        for leaf in unique_leaves:
            idx = np.where(leaf_ids == leaf)[0]
            chosen = idx[:2] if len(idx) >= 2 else idx
            gamma = leaf_gamma[leaf]
            print(f"Leaf node {leaf}, gamma = {gamma:.6f}")
            # contribution to F(x) for samples in this leaf after tree 1 is added
            for i in chosen:
                pred_value = F0 + eta * gamma
                print(
                    f"  sample {i}: x = {X[i]}, tree1 leaf prediction = gamma = {gamma:.6f}, "
                    f"updated model value after tree1 = F0 + eta*gamma = {pred_value:.6f}"
                )
        print()
    else:
        print()

    # Update model F(x)
    for i in range(n_samples):
        leaf = tree.apply(X[i].reshape(1, -1))[0]
        F[i] += eta * leaf_gamma[leaf]

# PART G
example_index = 0
x_example = X[example_index].reshape(1, -1)

# F(x) calculation for the example sample step-by-step
print(f"Example sample index: {example_index}")
print(f"Example sample features: {X[example_index]}")
print(f"True label: {y[example_index]}")
print(f"Start with F0 = {F0:.6f}")

F_example = F0
# go thru tree by tree and add contributions
for k in range(n_trees):
    tree = all_trees[k]
    leaf_gamma = all_leaf_gammas[k]
    leaf = tree.apply(x_example)[0]
    gamma = leaf_gamma[leaf]
    contribution = eta * gamma
    F_example += contribution
    print(
        f"Tree {k+1}: falls in leaf {leaf}, gamma = {gamma:.6f}, "
        f"contribution = eta*gamma = {contribution:.6f}, running F(x) = {F_example:.6f}"
    )

# final prediction
prob_example = sigmoid(F_example)
final_prediction = 1 if prob_example >= 0.5 else 0

print(f"Final raw score F(x) = {F_example:.6f}")
print(f"Final probability = sigmoid(F(x)) = {prob_example:.6f}")
print(f"Final predicted label = {final_prediction}")

