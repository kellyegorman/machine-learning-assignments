import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# class 1 training data
X_class1 = np.array([
    [-2, 0],
    [2, 0],
    [0, 2],
    [0, -2]
])

# class 2 training data
X_class2 = np.array([
    [-1, 0],
    [1, 0],
    [0, 1],
    [0, -1]
])

X = np.vstack((X_class1, X_class2))
y = np.array([1]*4 + [-1]*4)

# train gaussian svm
clf = SVC(kernel='rbf', gamma=1.0, C=1000)
clf.fit(X, y)

## PLOT

xx, yy = np.meshgrid(
    np.linspace(-3, 3, 500),
    np.linspace(-3, 3, 500)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = clf.decision_function(grid)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8,8))
# decision boundary on plot & plot points
plt.contourf(xx, yy, Z > 0, alpha=0.2)
plt.contour(xx, yy, Z, levels=[0], linewidths=2)
plt.scatter(X_class1[:,0], X_class1[:,1],
            color='blue', marker='o', s=100, label='Class 1')
plt.scatter(X_class2[:,0], X_class2[:,1],
            color='red', marker='x', s=100, label='Class 2')
plt.scatter(clf.support_vectors_[:,0],
            clf.support_vectors_[:,1],
            s=200,
            facecolors='none',
            edgecolors='k',
            linewidths=2,
            label='Support Vectors')

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Gaussian (RBF) Kernel SVM Decision Region')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')

plt.show()