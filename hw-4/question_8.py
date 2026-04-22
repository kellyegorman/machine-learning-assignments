import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# PART A
# Generate a random dataset with 20 samples. Each sample should have two input
# features and 1 output label (-1 or 1). (You can simply use the ones from the previous
# question, but make sure to relabel the outputs) Report the samples in the PDF of the solution
rng = np.random.default_rng(42)
n_samples = 20
# generate random features and binary labels (0 or 1)
X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 2))
# relabel 0 to -1 and keep 1 as is
y01 = rng.integers(0, 2, size=n_samples)
y = np.where(y01 == 0, -1, 1)

sample_ids = np.arange(n_samples)
# df for dataset
dataset_df = pd.DataFrame({
    "sample_id": sample_ids,
    "x1": np.round(X[:, 0], 4),
    "x2": np.round(X[:, 1], 4),
    "label": y
})

print("Random dataset with 20 samples")
print(dataset_df.to_string(index=False))

# convert sklearn predictions to {-1, 1}
def to_pm_one(pred):
    pred = np.asarray(pred)
    return np.where(pred >= 0, 1, -1)

# AdaBoost training 
n_learners = 10
n = len(y)
weights = np.ones(n) / n
learners = []
alphas = []

print("AdaBoost weak learners")

for j in range(n_learners):
    # train one decision stump using current sample weights
    stump = DecisionTreeClassifier(max_depth=1, random_state=100 + j)
    stump.fit(X, y, sample_weight=weights)
    preds = to_pm_one(stump.predict(X))
    # weighted classification error
    incorrect = (preds != y).astype(float)
    error = np.sum(weights * incorrect)
    # avoid division by zero or log of zero
    error = np.clip(error, 1e-10, 1 - 1e-10)
    # AdaBoost coefficient
    alpha = 0.5 * np.log((1 - error) / error)
    # update sample weights
    new_weights = weights * np.exp(-alpha * y * preds)
    new_weights = new_weights / np.sum(new_weights)
    # get split feature and threshold from sklearn tree
    split_feature = int(stump.tree_.feature[0])
    split_threshold = float(stump.tree_.threshold[0])

    learners.append(stump)
    alphas.append(alpha)

# details for each weak learner
    if j == 0:
        print("PART B: First weak learner (decision stump)")
        print(f"Split feature: x{split_feature + 1}")
        print(f"Threshold: {split_threshold:.6f}")
        print("PART C: Coefficient and updated weights after first learner")
        print(f"Weighted error: {error:.6f}")
        print(f"alpha_1: {alpha:.6f}")
        updated_weights_df = pd.DataFrame({
            "sample_id": sample_ids,
            "updated_weight": np.round(new_weights, 6)
        })
        print(updated_weights_df.to_string(index=False))
    else:
        print(f"\nWeak learner {j+1}")
        print(f"Split feature: x{split_feature + 1}")
        print(f"Threshold: {split_threshold:.6f}")
        print(f"alpha_{j+1}: {alpha:.6f}")

    weights = new_weights

print("PART D: Summary of all 10 weak learners")
summary_df = pd.DataFrame({
    "learner": [f"h{j+1}" for j in range(n_learners)],
    "split_feature": [f"x{int(stump.tree_.feature[0]) + 1}" for stump in learners],
    "threshold": [round(float(stump.tree_.threshold[0]), 6) for stump in learners],
    "alpha": [round(float(a), 6) for a in alphas]
})
print(summary_df.to_string(index=False))

# PART E
# show how you can predict using each weak learner and combine their results using the coefficients
example_index = 0
example_sample = X[example_index].reshape(1, -1)
true_label = int(y[example_index])

print("PART E: Prediction for one example sample")
print("Example sample index:", example_index)
print("Example sample features:", np.round(X[example_index], 4).tolist())
print("True label:", true_label)

running_sum = 0.0
# combine predictions from each weak learner using their coefficients
for j, (stump, alpha) in enumerate(zip(learners, alphas), start=1):
    pred = int(to_pm_one(stump.predict(example_sample))[0])
    # weight the prediction by its alpha coefficient
    weighted_vote = alpha * pred
    # add the weighted vote to the running sum
    running_sum += weighted_vote
    print(
        f"Learner {j}: prediction = {pred}, alpha_{j} = {alpha:.6f}, "
        f"alpha_{j} * prediction = {weighted_vote:.6f}"
    )

final_prediction = 1 if running_sum >= 0 else -1
print(f"Final weighted sum: {running_sum:.6f}")
print("Final AdaBoost output:", final_prediction)
