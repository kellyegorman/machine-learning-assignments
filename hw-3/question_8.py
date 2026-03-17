import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("hw-3/wine_dataset.csv")
X = df.drop("style", axis=1)

# dimension d
d = X.shape[1]
print("d =", d)

# standardize
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# covariance matrix
cov_matrix = np.cov(X_std, rowvar=False)
print("\nCovariance Matrix:\n", cov_matrix)

# eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

# get d/2 largest eigenvalues
idx = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[idx]
eigenvectors_sorted = eigenvectors[:, idx]
k = d // 2
top_eigenvalues = eigenvalues_sorted[:k]
top_eigenvectors = eigenvectors_sorted[:, :k]
print("\nTop eigenvalues:\n", top_eigenvalues)
print("\nTop eigenvectors:\n", top_eigenvectors)

#projection matrix
projection_matrix = top_eigenvectors
print("\nProjection Matrix:\n", projection_matrix)
