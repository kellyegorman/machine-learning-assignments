import numpy as np
import pandas as pd
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

# PART A 
# Generate a random dataset with 20 samples. 
# Each sample should have two input features and 1 output label (0 or 1). 
rng = np.random.default_rng(42)
n_samples = 20

# generate random features and binary labels
X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 2))
# labels are 0 or 1
y = rng.integers(0, 2, size=n_samples)
# create df for dataset
dataset_df = pd.DataFrame({
    "sample_id": np.arange(n_samples),
    "x1": np.round(X[:, 0], 4),
    "x2": np.round(X[:, 1], 4),
    "label": y
})

print("Random dataset with 20 samples")
print(dataset_df.to_string(index=False))

# PART B
# Generate 10 training datasets each of size 20, by sampling with 
# repetition and produce the output as a 10 × 20 matrix.
n_datasets = 10
bootstrap_indices = rng.integers(0, n_samples, size=(n_datasets, n_samples))

# bootstrap_indices is a 10 x 20 matrix where each row corresponds to a training dataset
bootstrap_df = pd.DataFrame(
    bootstrap_indices,
    index=[f"D{i+1}" for i in range(n_datasets)],
    columns=[f"S{j+1}" for j in range(n_samples)]
)

print("10 x 20 bootstrap index matrix")
print(bootstrap_df.to_string())

# PART C
# In the matrix above, highlight in bold all the entries 
# which are duplicated in each training dataset

# boolean mask to identify duplicates in each row
duplicate_mask = np.zeros_like(bootstrap_indices, dtype=bool)

# check for duplicates in each row of training dataset
for i in range(n_datasets):
    counts = Counter(bootstrap_indices[i])
    duplicate_mask[i] = [counts[val] > 1 for val in bootstrap_indices[i]]

print("Bootstrap matrix with duplicated entries marked ** **")
# print the matrix with duplicates highlighted by ****
for i in range(n_datasets):
    row_out = []
    for j in range(n_samples):
        value = bootstrap_indices[i, j]
        if duplicate_mask[i, j]:
            row_out.append(f"**{value}**")
        else:
            row_out.append(str(value))
    print(f"D{i+1}: " + ", ".join(row_out))

# PART D
# Train 10 classifier models (you can choose your own) on the 10 datasets
models = []

# train a decision tree classifier on each bootstrap dataset
for i in range(n_datasets):
    idx = bootstrap_indices[i]
    X_boot = X[idx]
    y_boot = y[idx]

# use a simple decision tree classifier for demonstration
    model = DecisionTreeClassifier(random_state=100 + i)
    model.fit(X_boot, y_boot)
    models.append(model)

print("Trained 10 decision tree classifier models.")

# PART E
# For an example sample, show how you can use majority voting to 
# combine the results and produce the final output.

# take the first sample from the original dataset as an examples
example_sample = X[0].reshape(1, -1)
example_true_label = y[0]

# get predictions from each of the 10 models for the example sample
individual_predictions = [int(model.predict(example_sample)[0]) for model in models]
vote_counts = Counter(individual_predictions)
final_prediction = 1 if vote_counts[1] > vote_counts[0] else 0

print("Majority voting example")
print("Example sample (x1, x2):", np.round(example_sample[0], 4).tolist())
print("True label:", int(example_true_label))
print("Predictions from 10 models:", individual_predictions)
print("Votes for class 0:", vote_counts[0])
print("Votes for class 1:", vote_counts[1])
print("Final output after majority voting:", final_prediction)
