import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv("hw-2/wine_dataset.csv")

# red wine -> 0 label, white wine -> 1 label
data["style"] = data["style"].map({"red": 0, "white": 1})
X = data.drop("style", axis=1)
y = data["style"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=1
)

# Logistic regression, testing out different C values
lr = LogisticRegression(C=65.0, max_iter=5000) 
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
# different outputs: 
#   C = 1.0 -> 0.985
#   C = 5.0 -> 0.987
#   C = 10.0 -> 0.987
#   C = 50.0 -> 0.988
#   C = 65.0 -> 0.989 *** Best C value
#   C = 80.0 -> 0.989 
#   C = 100.0 -> 0.989
#   C = 150.0 -> 0.988


print("LR accurarcy is ", lr_accuracy)

# SVM, change kernel, C, gamma to test 
#    try for kernel: 'linear', 'poly', or 'rbf'
#    try for C: different values
#    try for gamma: differecent values, but only for rbf
svm = SVC(
    kernel='linear',  
    C=65.0,         
    gamma=0.01     
)
# different kernel
    # rbf -> 0.965 (C=65, gamma=0.01) (fastest)
    # linear -> 0.989 (C=65)
    # poly -> 0.984 (C=65) (took much longer)
# different C:
    # 65 -> 0.965
    # 80 -> 0.964
# different gamma:
    # 0.1 -> 0.938
    # 0.5 -> 0.840
    # 0.8 -> 0.809
    # 0.01 -> 0.939
    # 0.001 -> 0.936

svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

print("SVM accuracy is ", svm_accuracy)