import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("hw-3/wine_dataset.csv")
X = df.drop("style", axis=1)
y = df["style"]

# standardize
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
d = X_std.shape[1]

# mean vectors for classes
class_labels = np.unique(y)
means = {}
for label in class_labels:
    X_c = X_std[y == label]
    means[label] = np.mean(X_c, axis=0)
    print(f"\nMean vector for class {label}:\n", means[label])

# overall mean
overall_mean = np.mean(X_std, axis=0)

# SW/SB
SW = np.zeros((d, d))
SB = np.zeros((d, d))

for label in class_labels:
    X_c = X_std[y == label]
    mean_vec = means[label]
    for x in X_c:
        diff = (x - mean_vec).reshape(d, 1)
        SW += diff @ diff.T
    n_c = X_c.shape[0]
    mean_diff = (mean_vec - overall_mean).reshape(d, 1)
    SB += n_c * (mean_diff @ mean_diff.T)

print("\nSW:\n", SW)
print("\nSB:\n", SB)

# eigenvalues/vectors of (SW^-1)*(SB)
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(SW).dot(SB))

print("\nEigenvalues:\n", eigvals)
print("\nEigenvectors:\n", eigvecs)

# top d/2
idx = np.argsort(eigvals)[::-1]
eigvals_sorted = eigvals[idx]
eigvecs_sorted = eigvecs[:, idx]
k = d // 2
top_vals = eigvals_sorted[:k]
top_vecs = eigvecs_sorted[:, :k]

print("\nTop eigenvalues:\n", top_vals)
print("\nTop eigenvectors:\n", top_vecs)

# proj matrix
W = top_vecs
print("\nProjection Matrix:\n", W)
